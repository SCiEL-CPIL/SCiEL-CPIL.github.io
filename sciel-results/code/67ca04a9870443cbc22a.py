import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s

    n = 16
    points = list(range(n))
    all_triples = list(combinations(points, 3))
    
    # Precompute pair keys for each triple
    triple_pairs = {}
    triple_masks = {}
    for t in all_triples:
        pairs = set()
        mask = 0
        for i in range(3):
            for j in range(i+1, 3):
                p1, p2 = t[i], t[j]
                key = (p1, p2) if p1 < p2 else (p2, p1)
                pairs.add(key)
        triple_pairs[t] = pairs
        for p in t:
            mask |= (1 << p)
        triple_masks[t] = mask

    # Precompute popcounts and masks by popcount for fast alpha search
    max_mask = (1 << n) - 1
    popcounts = [0] * (1 << n)
    masks_by_popcount = [[] for _ in range(n + 1)]
    for mask in range(1 << n):
        pc = bin(mask).count('1')
        popcounts[mask] = pc
        masks_by_popcount[pc].append(mask)

    def get_alpha_and_set(triples):
        """
        Finds the largest independent set (max popcount mask containing no triple).
        Iterates popcounts from high to low.
        """
        if not triples:
            return n, max_mask

        t_masks = [triple_masks[t] for t in triples]
        
        # Check popcounts from n down to 0
        for k in range(n, -1, -1):
            for mask in masks_by_popcount[k]:
                is_independent = True
                for tm in t_masks:
                    if (mask & tm) == tm:
                        is_independent = False
                        break
                if is_independent:
                    return k, mask
        
        return 0, 0

    def is_valid_addition(triple, used_pairs):
        """Check if triple can be added without violating linearity."""
        for p in triple_pairs[triple]:
            if p in used_pairs:
                return False
        return True

    def greedy_construct():
        triples = []
        used_pairs = set()
        shuffled = all_triples[:]
        random.shuffle(shuffled)
        for t in shuffled:
            if is_valid_addition(t, used_pairs):
                triples.append(t)
                used_pairs.update(triple_pairs[t])
        return triples

    def local_search_lns(triples, deadline):
        """
        Large Neighborhood Search.
        """
        if not triples:
            return triples, n, max_mask

        current_triples = triples[:]
        current_alpha, current_is_mask = get_alpha_and_set(current_triples)
        best_alpha = current_alpha
        best_triples = current_triples[:]
        best_is_mask = current_is_mask

        used_pairs = set()
        for t in current_triples:
            used_pairs.update(triple_pairs[t])

        temperature = 1.0
        cooling = 0.99
        destroy_rate = 0.2
        max_steps = 1000

        for step in range(max_steps):
            if time.time() > deadline:
                break

            if step % 50 == 0:
                temperature *= cooling

            # 1. Destroy: Remove a random subset of triples
            num_remove = max(1, int(len(current_triples) * destroy_rate))
            if num_remove >= len(current_triples):
                num_remove = len(current_triples) - 1 if len(current_triples) > 1 else 0

            removed_triples = random.sample(current_triples, num_remove) if num_remove > 0 else []
            removed_set = set(removed_triples)
            
            # Calculate used_pairs for remaining triples
            new_used_pairs = set()
            for t in current_triples:
                if t not in removed_set:
                    new_used_pairs.update(triple_pairs[t])
            
            remaining_triples = [t for t in current_triples if t not in removed_set]

            # 2. Repair: Greedily add triples
            # Candidates: all triples not in remaining_triples and valid against new_used_pairs
            # We prioritize triples that hit the current IS mask
            candidates = []
            for t in all_triples:
                if t in remaining_triples:
                    continue
                if is_valid_addition(t, new_used_pairs):
                    candidates.append(t)
            
            # Sort candidates by overlap with current_is_mask (descending)
            # This targets breaking the current independent set
            scored_cands = []
            for t in candidates:
                overlap = bin(triple_masks[t] & current_is_mask).count('1')
                scored_cands.append((overlap, t))
            
            scored_cands.sort(key=lambda x: -x[0])
            
            new_triples = remaining_triples[:]
            new_used_pairs = new_used_pairs.copy()
            
            for _, t in scored_cands:
                # Check if still valid (it should be, as we only add, but used_pairs doesn't change in a way that invalidates previous adds? 
                # Wait, adding a triple uses pairs. It doesn't free pairs. So if it was valid against new_used_pairs, it remains valid.
                # However, we must ensure we don't add duplicates.
                # Since we iterate unique candidates, this is fine.
                new_triples.append(t)
                new_used_pairs.update(triple_pairs[t])
                
                # Optional: Stop adding if we hit a time limit or if we've added enough?
                # Let's add all valid ones? No, that might increase size too much.
                # But for LNS, we usually try to restore to a similar size or just maximize quality.
                # Let's just add all valid ones for now, as more triples generally helps reduce alpha.
                # Actually, adding too many might be redundant. But linear constraint prevents overlap > 1.
                # Max size is bounded.
                
            new_alpha, new_is_mask = get_alpha_and_set(new_triples)
            
            # 3. Acceptance
            delta = new_alpha - current_alpha
            
            if delta < 0:
                current_triples = new_triples
                current_alpha = new_alpha
                current_is_mask = new_is_mask
                used_pairs = new_used_pairs
                
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_triples = current_triples[:]
                    best_is_mask = current_is_mask
                    
                if best_alpha <= 4:
                    break
            else:
                # Accept worse with probability exp(-delta / temperature)
                if temperature > 0.01:
                    prob = pow(2.718281828, -delta / max(temperature, 0.01))
                    if random.random() < prob:
                        current_triples = new_triples
                        current_alpha = new_alpha
                        current_is_mask = new_is_mask
                        used_pairs = new_used_pairs

        return best_triples, best_alpha, best_is_mask

    # Main
    best_triples = []
    best_alpha = n
    best_is_mask = max_mask

    # Multiple restarts
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break

        triples = greedy_construct()
        if not triples:
            continue

        triples, alpha, is_mask = local_search_lns(triples, deadline - 5)

        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples[:]
            best_is_mask = is_mask

        if best_alpha <= 4:
            break

    return [list(t) for t in best_triples]

if __name__ == "__main__":
    triples = construct()
    print(f"Found {len(triples)} triples")
    used_pairs = set()
    valid = True
    for t in triples:
        for i in range(3):
            for j in range(i+1, 3):
                p1, p2 = t[i], t[j]
                key = (p1, p2) if p1 < p2 else (p2, p1)
                if key in used_pairs:
                    print(f"INVALID: Shared pair {key} in {t}")
                    valid = False
                used_pairs.add(key)
    if valid:
        print("Valid linear system")
