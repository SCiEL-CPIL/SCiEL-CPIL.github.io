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
    for t in all_triples:
        pairs = set()
        for i in range(3):
            for j in range(i+1, 3):
                p1, p2 = t[i], t[j]
                key = (p1, p2) if p1 < p2 else (p2, p1)
                pairs.add(key)
        triple_pairs[t] = pairs

    # Precompute bitmask for each triple
    triple_masks = {}
    for t in all_triples:
        m = 0
        for p in t:
            m |= (1 << p)
        triple_masks[t] = m
            
    # For each point, list of triples containing it
    point_to_triples = {p: [] for p in points}
    for t in all_triples:
        for p in t:
            point_to_triples[p].append(t)

    def get_alpha_fast(triples):
        """
        Fast branch-and-bound independence number.
        Returns the size of the largest subset of points containing no triple.
        """
        if not triples:
            return n

        t_masks = [triple_masks[t] for t in triples]
        num_t = len(t_masks)

        # For each point, bitmask of triples containing it
        # This helps in quick checking if adding a point creates a triple
        # Actually, for independence, we need to check if the *subset* contains a triple.
        # A subset S contains a triple T if (mask_S & mask_T) == mask_T.
        
        # Sort points by degree (number of triples containing them) descending
        # Points in more triples are "more constrained"
        # We want to decide on these first? Or last?
        # Usually, high degree points are more likely to be excluded in a small IS?
        # No, if we want to MAXIMIZE IS, we want to include points.
        # If a point is in many triples, including it makes it harder to include others.
        # So we should decide on them early to prune branches where they are included but conflict?
        # Let's try sorting by degree descending.
        
        degrees = [0] * n
        for tm in t_masks:
            for p in range(n):
                if tm & (1 << p):
                    degrees[p] += 1
        
        order = sorted(range(n), key=lambda p: degrees[p], reverse=True)
        
        best = [0]
        
        # Greedy heuristic to find a good initial lower bound
        # Sort by degree ascending? Include least constrained first?
        # Or random? Let's do a few random greedy passes.
        for _ in range(5):
            shuffled_order = order[:]
            random.shuffle(shuffled_order)
            greedy_mask = 0
            greedy_count = 0
            for p in shuffled_order:
                new_mask = greedy_mask | (1 << p)
                ok = True
                for tm in t_masks:
                    if (new_mask & tm) == tm:
                        ok = False
                        break
                if ok:
                    greedy_mask = new_mask
                    greedy_count += 1
            if greedy_count > best[0]:
                best[0] = greedy_count

        # B&B
        # State: (idx, mask, count)
        # idx: index in order
        stack = [(0, 0, 0)]
        
        while stack:
            pos, mask, count = stack.pop()
            # Upper bound: current count + remaining points
            if count + (n - pos) <= best[0]:
                continue
            if pos == n:
                if count > best[0]:
                    best[0] = count
                continue

            p = order[pos]
            
            # Option 1: Don't include p
            stack.append((pos + 1, mask, count))

            # Option 2: Include p
            new_mask = mask | (1 << p)
            # Check if new_mask contains any triple
            valid = True
            for tm in t_masks:
                if (new_mask & tm) == tm:
                    valid = False
                    break
            
            if valid:
                # Prune if even adding all remaining doesn't beat best
                if count + 1 + (n - pos - 1) > best[0]:
                    stack.append((pos + 1, new_mask, count + 1))

        return best[0]

    def get_alpha_with_set(triples):
        """Returns (alpha, max_independent_set) where max_independent_set is a set of points."""
        if not triples:
            return n, set(points)

        t_masks = [triple_masks[t] for t in triples]
        
        degrees = [0] * n
        for tm in t_masks:
            for p in range(n):
                if tm & (1 << p):
                    degrees[p] += 1
        
        order = sorted(range(n), key=lambda p: degrees[p], reverse=True)
        
        best = [0]
        best_set = [set()]

        # Greedy first
        for _ in range(5):
            shuffled_order = order[:]
            random.shuffle(shuffled_order)
            greedy_mask = 0
            greedy_count = 0
            for p in shuffled_order:
                new_mask = greedy_mask | (1 << p)
                ok = True
                for tm in t_masks:
                    if (new_mask & tm) == tm:
                        ok = False
                        break
                if ok:
                    greedy_mask = new_mask
                    greedy_count += 1
            if greedy_count > best[0]:
                best[0] = greedy_count
                best_set[0] = set(p for p in range(n) if greedy_mask & (1 << p))

        # B&B with set tracking
        # To save memory/time, we can reconstruct the set at the end or track it.
        # Tracking it in the stack is expensive. Let's just track the mask and reconstruct if we find a new best.
        # But we need the set for the local search.
        # Let's modify B&B to return the best mask, then convert to set.
        
        best_mask = [0]
        
        stack = [(0, 0, 0)]
        while stack:
            pos, mask, count = stack.pop()
            if count + (n - pos) <= best[0]:
                continue
            if pos == n:
                if count > best[0]:
                    best[0] = count
                    best_mask[0] = mask
                continue

            p = order[pos]
            # Don't include
            stack.append((pos + 1, mask, count))

            # Include
            new_mask = mask | (1 << p)
            valid = True
            for tm in t_masks:
                if (new_mask & tm) == tm:
                    valid = False
                    break
            if valid and count + 1 + (n - pos - 1) > best[0]:
                stack.append((pos + 1, new_mask, count + 1))

        # Convert best_mask to set
        final_set = set()
        m = best_mask[0]
        for p in range(n):
            if m & (1 << p):
                final_set.add(p)
        
        return best[0], final_set

    def is_valid_addition(triple, used_pairs):
        """Check if triple can be added without violating linearity."""
        for p in triple_pairs[triple]:
            if p in used_pairs:
                return False
        return True

    def greedy_construct_backtrack():
        """
        Construct a large linear system using backtracking.
        We prioritize adding triples that cover many new pairs? 
        Or just random with backtracking if stuck?
        Let's try a simple recursive backtracking that tries to add as many triples as possible.
        """
        triples = []
        used_pairs = set()
        
        def backtrack(remaining_triples):
            if time.time() > deadline - 10:
                return
            if not remaining_triples:
                return
            
            # Heuristic: pick a random triple from remaining
            # Or pick one that conflicts with fewest existing triples?
            # For simplicity, random selection with backtracking.
            
            # To avoid deep recursion, we limit depth or use iterative?
            # Max triples in linear system on 16 points:
            # Each triple uses 3 pairs. Total pairs 120. Max 40 triples.
            # Depth 40 is fine.
            
            idx = random.randint(0, len(remaining_triples) - 1)
            t = remaining_triples[idx]
            
            if is_valid_addition(t, used_pairs):
                triples.append(t)
                used_pairs.update(triple_pairs[t])
                new_remaining = [x for x in remaining_triples if x != t]
                backtrack(new_remaining)
                # Backtrack
                triples.pop()
                used_pairs.difference_update(triple_pairs[t])
            
            # If we didn't add it, we just skip it? 
            # This is not a full search for max packing, just a greedy with randomization.
            # To be more effective, we should try to add *some* triple.
            # If we can't add the chosen one, we should try another.
            # This simple implementation might terminate early if it picks bad triples.
            
            # Better: Shuffle and pick first valid?
            pass

        # Simpler robust greedy:
        # 1. Shuffle all triples
        # 2. Iterate, add if valid
        # This is what the original did. 
        # Let's stick to the original greedy but maybe run it multiple times and keep the one with largest size?
        # Actually, the size of the system matters for alpha. More triples -> smaller alpha potentially.
        # So we want a *maximal* or *maximum* packing.
        
        best_size = 0
        best_system = []
        
        for _ in range(20):
            if time.time() > deadline - 10:
                break
            sys = []
            used = set()
            shuffled = all_triples[:]
            random.shuffle(shuffled)
            for t in shuffled:
                if is_valid_addition(t, used):
                    sys.append(t)
                    used.update(triple_pairs[t])
            if len(sys) > best_size:
                best_size = len(sys)
                best_system = sys[:]
                
        return best_system

    def local_search_targeted(triples, deadline):
        """
        Local search that specifically targets the independent set.
        """
        if not triples:
            return triples, n, set(points)

        current_alpha, current_iset = get_alpha_with_set(triples)
        best_alpha = current_alpha
        best_triples = triples[:]
        best_iset = current_iset

        used_pairs = set()
        for t in triples:
            used_pairs.update(triple_pairs[t])

        current = triples[:]
        cur_alpha = current_alpha
        cur_iset = current_iset

        # Precompute candidates that are NOT in current
        # We will maintain a list of valid candidates dynamically
        
        max_steps = 5000
        no_improve_count = 0
        stagnation_limit = 200

        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            
            if no_improve_count > stagnation_limit:
                # Restart or perturb
                # Perturb: remove 2 random triples, re-greedy
                if len(current) > 4:
                    current = current[:]
                    used_pairs = set()
                    for t in current:
                        used_pairs.update(triple_pairs[t])
                    
                    # Remove 2
                    idx1 = random.randint(0, len(current)-1)
                    idx2 = random.randint(0, len(current)-2)
                    if idx2 >= idx1: idx2 += 1
                    
                    t1 = current.pop(idx1)
                    t2 = current.pop(idx2)
                    used_pairs.difference_update(triple_pairs[t1])
                    used_pairs.difference_update(triple_pairs[t2])
                    
                    # Re-greedy add
                    shuffled = all_triples[:]
                    random.shuffle(shuffled)
                    for t in shuffled:
                        if t in current: continue
                        if is_valid_addition(t, used_pairs):
                            current.append(t)
                            used_pairs.update(triple_pairs[t])
                    
                    cur_alpha, cur_iset = get_alpha_with_set(current)
                    no_improve_count = 0
                else:
                    break

            # Identify candidates for addition
            # A candidate is valid if it doesn't share a pair with any triple in current
            # And it is not already in current
            
            # To speed up, we can just check all triples not in current
            # But we want to prioritize those that hit the IS
            
            iset_list = list(cur_iset)
            
            # Find triples that intersect IS in >= 2 points
            # These are the most promising for reducing alpha
            promising = []
            other_valid = []
            
            current_set = set(current)
            
            for t in all_triples:
                if t in current_set:
                    continue
                if not is_valid_addition(t, used_pairs):
                    continue
                
                overlap = len(set(t) & cur_iset)
                if overlap >= 2:
                    promising.append(t)
                else:
                    other_valid.append(t)
            
            if not promising and not other_valid:
                # No moves possible?
                # Maybe we need to swap?
                # For now, if no adds, break or perturb
                no_improve_count += 1
                continue

            # Choose a move
            # 70% chance pick from promising, 30% from other_valid
            if promising and (not other_valid or random.random() < 0.7):
                t_add = random.choice(promising)
            else:
                t_add = random.choice(other_valid)
            
            # Try adding t_add
            new_triples = current + [t_add]
            new_used_pairs = used_pairs | triple_pairs[t_add]
            
            new_alpha, new_iset = get_alpha_with_set(new_triples)
            
            if new_alpha < cur_alpha:
                current = new_triples
                cur_alpha = new_alpha
                cur_iset = new_iset
                used_pairs = new_used_pairs
                no_improve_count = 0
                
                if cur_alpha < best_alpha:
                    best_alpha = cur_alpha
                    best_triples = current[:]
                    best_iset = cur_iset
                    
                if best_alpha <= 4:
                    break
            else:
                no_improve_count += 1
                # Optionally accept worse with low probability? 
                # Since we are targeting reduction, maybe don't accept worse unless stuck?
                # Let's accept worse with small probability to escape local optima
                if random.random() < 0.1:
                    current = new_triples
                    cur_alpha = new_alpha
                    cur_iset = new_iset
                    used_pairs = new_used_pairs
                    no_improve_count = 0

        return best_triples, best_alpha, best_iset

    # Main
    best_triples = []
    best_alpha = n
    best_iset = set(points)

    # Phase 1: Get a good initial packing
    initial_triples = greedy_construct_backtrack()
    
    # Phase 2: Local search on this packing
    if initial_triples:
        triples, alpha, iset = local_search_targeted(initial_triples, deadline - 10)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples[:]
            best_iset = iset

    # Phase 3: Multiple restarts with different initial packings
    # We have ~100s left. 
    # Let's do 10 restarts.
    
    for i in range(10):
        if time.time() > deadline - 5:
            break
        
        # Generate a new random packing
        sys = []
        used = set()
        shuffled = all_triples[:]
        random.shuffle(shuffled)
        for t in shuffled:
            if is_valid_addition(t, used):
                sys.append(t)
                used.update(triple_pairs[t])
        
        if not sys:
            continue
            
        triples, alpha, iset = local_search_targeted(sys, deadline - 5)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples[:]
            best_iset = iset
            
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
