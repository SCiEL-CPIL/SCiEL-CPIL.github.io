import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n = 16
    points = list(range(n))
    
    # All possible triples
    all_triples = list(combinations(points, 3))
    # Index each triple by its 3 points
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    # For each point, which triples contain it
    point_triples = {p: [] for p in points}
    for t in all_triples:
        for p in t:
            point_triples[p].append(t)
    # For each pair, which triple contains it (unique if linear)
    pair_triple = {}
    for t in all_triples:
        for i in range(3):
            for j in range(i+1, 3):
                p1, p2 = t[i], t[j]
                key = (p1, p2) if p1 < p2 else (p2, p1)
                pair_triple[key] = t
    
    def get_alpha_fast(triples):
        """Compute independence number using branch-and-bound on hitting set."""
        if not triples:
            return n

        # Build bitmask for each triple
        t_masks = []
        for t in triples:
            m = 0
            for p in t:
                m |= (1 << p)
            t_masks.append(m)

        # alpha = n - minimum hitting set size
        # Find minimum hitting set using branch and bound

        num_triples = len(t_masks)
        if num_triples == 0:
            return n

        # Upper bound: all points that appear
        all_pts = 0
        for tm in t_masks:
            all_pts |= tm
        upper_bound = bin(all_pts).count('1')
        best = [upper_bound]

        # Precompute for each point, bitmask of triples it hits
        point_hit_mask = [0] * n
        for idx, tm in enumerate(t_masks):
            for p in range(n):
                if tm & (1 << p):
                    point_hit_mask[p] |= (1 << idx)

        def solve(remaining_mask, cur_size):
            if cur_size >= best[0]:
                return
            if remaining_mask == 0:
                best[0] = cur_size
                return

            # Pick a triple from remaining
            first_bit = remaining_mask & (-remaining_mask)
            first_idx = first_bit.bit_length() - 1
            first_triple = t_masks[first_idx]

            # Branch on each point in this triple
            for p in range(n):
                if not (first_triple & (1 << p)):
                    continue

                new_remaining = remaining_mask & ~point_hit_mask[p]
                solve(new_remaining, cur_size + 1)
                if best[0] <= cur_size + 1:
                    return

        solve((1 << num_triples) - 1, 0)
        return n - best[0]
    
    def is_linear(triples):
        """Check if any pair is repeated."""
        seen_pairs = set()
        for t in triples:
            for i in range(3):
                for j in range(i+1, 3):
                    p1, p2 = t[i], t[j]
                    key = (p1, p2) if p1 < p2 else (p2, p1)
                    if key in seen_pairs:
                        return False
                    seen_pairs.add(key)
        return True
    
    def local_search(triples, deadline):
        """Try to improve alpha by swapping triples."""
        if not triples:
            return triples, get_alpha_fast(triples)
        
        current_alpha = get_alpha_fast(triples)
        best_alpha = current_alpha
        best_triples = triples[:]
        
        # Precompute used pairs
        used_pairs = set()
        for t in triples:
            for i in range(3):
                for j in range(i+1, 3):
                    p1, p2 = t[i], t[j]
                    key = (p1, p2) if p1 < p2 else (p2, p1)
                    used_pairs.add(key)
        
        # Candidate triples to swap in
        candidates = []
        for t in all_triples:
            if t in triples:
                continue
            # Check if t can be added (no pair conflict)
            can_add = True
            for i in range(3):
                for j in range(i+1, 3):
                    p1, p2 = t[i], t[j]
                    key = (p1, p2) if p1 < p2 else (p2, p1)
                    if key in used_pairs:
                        can_add = False
                        break
                if not can_add:
                    break
            if can_add:
                candidates.append(t)
        
        if not candidates:
            return best_triples, best_alpha
        
        # Try swaps: remove one triple, add one candidate
        # Simulated annealing style
        temperature = 0.5
        cooling = 0.995
        current = triples[:]
        cur_alpha = current_alpha
        
        max_steps = 8000
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            if step % 50 == 0:
                temperature *= cooling
            
            # Pick a random triple to remove
            idx = random.randint(0, len(current) - 1)
            removed = current[idx]
            
            # Pick a random candidate to add
            # Filter candidates that are compatible with current minus removed
            # Recompute used_pairs after removing
            new_used_pairs = set()
            for t in current:
                if t == removed:
                    continue
                for i in range(3):
                    for j in range(i+1, 3):
                        p1, p2 = t[i], t[j]
                        key = (p1, p2) if p1 < p2 else (p2, p1)
                        new_used_pairs.add(key)
            
            # Find candidates that can be added
            valid_cands = []
            for t in candidates:
                if t == removed:
                    continue
                can_add = True
                for i in range(3):
                    for j in range(i+1, 3):
                        p1, p2 = t[i], t[j]
                        key = (p1, p2) if p1 < p2 else (p2, p1)
                        if key in new_used_pairs:
                            can_add = False
                            break
                    if not can_add:
                        break
                if can_add:
                    valid_cands.append(t)
            
            if not valid_cands:
                continue
            
            added = random.choice(valid_cands)
            new_triples = current[:idx] + current[idx+1:] + [added]
            new_alpha = get_alpha_fast(new_triples)
            
            # Accept if better, or with probability exp(-delta/temperature)
            delta = new_alpha - cur_alpha
            if delta < 0:
                current = new_triples
                cur_alpha = new_alpha
                if cur_alpha < best_alpha:
                    best_alpha = cur_alpha
                    best_triples = current[:]
            else:
                if random.random() < pow(2.718281828, -delta / max(temperature, 0.01)):
                    current = new_triples
                    cur_alpha = new_alpha
        
        # Final polish: try to add more triples if possible
        while time.time() < deadline:
            used_pairs_final = set()
            for t in best_triples:
                for i in range(3):
                    for j in range(i+1, 3):
                        p1, p2 = t[i], t[j]
                        key = (p1, p2) if p1 < p2 else (p2, p1)
                        used_pairs_final.add(key)

            added_any = False
            for t in all_triples:
                if t in best_triples:
                    continue
                can_add = True
                for i in range(3):
                    for j in range(i+1, 3):
                        p1, p2 = t[i], t[j]
                        key = (p1, p2) if p1 < p2 else (p2, p1)
                        if key in used_pairs_final:
                            can_add = False
                            break
                    if not can_add:
                        break
                if can_add:
                    test_triples = best_triples + [t]
                    test_alpha = get_alpha_fast(test_triples)
                    if test_alpha <= best_alpha:
                        best_triples = test_triples
                        best_alpha = test_alpha
                        used_pairs_final = set()
                        for tt in best_triples:
                            for i in range(3):
                                for j in range(i+1, 3):
                                    p1, p2 = tt[i], tt[j]
                                    key = (p1, p2) if p1 < p2 else (p2, p1)
                                    used_pairs_final.add(key)
                        added_any = True
                        break

            if not added_any:
                break

        return best_triples, best_alpha
    
    # Main: multiple random restarts with local search
    best_triples = []
    best_alpha = n
    
    # First, do a greedy construction to get a decent starting point
    def greedy_construct_smart():
        """Greedy construction that prioritizes triples covering high-degree points."""
        triples = []
        used_pairs = set()
        point_count = [0] * n

        shuffled = all_triples[:]
        random.shuffle(shuffled)

        for t in shuffled:
            valid = True
            for i in range(3):
                for j in range(i+1, 3):
                    p1, p2 = t[i], t[j]
                    key = (p1, p2) if p1 < p2 else (p2, p1)
                    if key in used_pairs:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                triples.append(t)
                for i in range(3):
                    for j in range(i+1, 3):
                        p1, p2 = t[i], t[j]
                        key = (p1, p2) if p1 < p2 else (p2, p1)
                        used_pairs.add(key)
                for p in t:
                    point_count[p] += 1

        return triples
    
    # Run greedy + local search multiple times
    num_restarts = 40
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Greedy start
        triples = greedy_construct_smart()
        if not triples:
            continue
        
        # Local search
        triples, alpha = local_search(triples, deadline)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples[:]
        
        if best_alpha <= 4:
            break
    
    # If we still have time, try more restarts
    while time.time() < deadline - 10:
        triples = greedy_construct_smart()
        if not triples:
            continue
        triples, alpha = local_search(triples, deadline)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples[:]
    
    # Convert to list of lists
    return [list(t) for t in best_triples]

if __name__ == "__main__":
    triples = construct()
    print(f"Found {len(triples)} triples")
    # Verify linearity
    used_pairs = set()
    for t in triples:
        for i in range(3):
            for j in range(i+1, 3):
                p1, p2 = t[i], t[j]
                key = (p1, p2) if p1 < p2 else (p2, p1)
                if key in used_pairs:
                    print("INVALID: Shared pair", t)
                    break
                used_pairs.add(key)
    else:
        print("Valid linear system")
