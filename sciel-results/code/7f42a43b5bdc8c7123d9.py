import time
import random
import itertools
import numpy as np
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n = 16
    points = list(range(n))
    all_triples = list(itertools.combinations(points, 3))
    num_triples = len(all_triples)
    
    # Precompute masks for all triples
    triple_masks = [0] * num_triples
    for i, t in enumerate(all_triples):
        triple_masks[i] = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        
    # Precompute pair masks for linearity check
    tpair = [0] * num_triples
    for i, (a, b, c) in enumerate(all_triples):
        pm = 0
        for x, y in ((a, b), (a, c), (b, c)):
            if x > y:
                x, y = y, x
            pm |= 1 << (x * n + y)
        tpair[i] = pm
        
    # Precompute popcount array for all 2^16 masks
    pop_arr = np.zeros(1 << n, dtype=np.int8)
    for i in range(1, 1 << n):
        pop_arr[i] = pop_arr[i >> 1] + (i & 1)
        
    # Precompute subsets array for vectorized coverage check
    subsets = np.arange(1 << n, dtype=np.int32)
    
    def compute_alpha_fast(triple_indices):
        """
        Compute the independence number of the hypergraph defined by triple_indices.
        Returns the size of the largest subset of points containing no triple.
        Uses vectorized NumPy operations for speed.
        """
        if not triple_indices:
            return n
        
        covered = np.zeros(1 << n, dtype=bool)
        
        for idx in triple_indices:
            t_mask = triple_masks[idx]
            covered |= (subsets & t_mask) == t_mask
            
        max_alpha = np.max(pop_arr[~covered])
        return int(max_alpha)
    
    def greedy_packing(order, existing_set=None, existing_pair_union=0):
        """
        Greedily add triples in the given order if they are linear with existing ones.
        """
        if existing_set is None:
            existing_set = set()
        
        sys_set = set(existing_set)
        pair_union = existing_pair_union
        
        for idx in order:
            if idx not in sys_set and (tpair[idx] & pair_union) == 0:
                sys_set.add(idx)
                pair_union |= tpair[idx]
                
        return list(sys_set), pair_union
    
    # Helper to get valid candidates
    def get_valid_candidates(current_set, current_pair_union):
        valid = []
        for idx in range(num_triples):
            if idx not in current_set and (tpair[idx] & current_pair_union) == 0:
                valid.append(idx)
        return valid
    
    # Initial solution: multiple random restarts
    best_alpha = n
    global_best_packing = []
    
    for restart in range(5):
        order = list(range(num_triples))
        random.shuffle(order)
        packing, pair_union = greedy_packing(order)
        alpha = compute_alpha_fast(packing)
        
        if alpha < best_alpha:
            best_alpha = alpha
            global_best_packing = packing
    
    # Start local search from best packing
    current_packing = list(global_best_packing)
    current_set = set(current_packing)
    current_pair_union = 0
    for idx in current_packing:
        current_pair_union |= tpair[idx]
    current_alpha = best_alpha
    
    # Precompute point counts for each triple index
    triple_points = [set(all_triples[i]) for i in range(num_triples)]
    
    while time.time() < deadline:
        r = random.random()
        
        if r < 0.4:
            # Try to add a random valid triple
            candidates = get_valid_candidates(current_set, current_pair_union)
            if candidates:
                cand = random.choice(candidates)
                new_packing = current_packing + [cand]
                new_alpha = compute_alpha_fast(new_packing)
                if new_alpha <= current_alpha:
                    current_packing = new_packing
                    current_set.add(cand)
                    current_pair_union |= tpair[cand]
                    current_alpha = new_alpha
                    if current_alpha < best_alpha:
                        best_alpha = current_alpha
                        global_best_packing = list(current_packing)
                        
        elif r < 0.7:
            # Swap: remove one, add one
            if not current_packing:
                continue
            rem = random.choice(current_packing)
            
            # Get valid candidates excluding those that conflict with current_set - {rem}
            temp_pair_union = 0
            for idx in current_packing:
                if idx != rem:
                    temp_pair_union |= tpair[idx]
            
            candidates = get_valid_candidates(current_set - {rem}, temp_pair_union)
            if candidates:
                cand = random.choice(candidates)
                new_packing = [x for x in current_packing if x != rem] + [cand]
                new_alpha = compute_alpha_fast(new_packing)
                
                if new_alpha <= current_alpha:
                    current_packing = new_packing
                    current_set.discard(rem)
                    current_set.add(cand)
                    current_pair_union = 0
                    for idx in current_packing:
                        current_pair_union |= tpair[idx]
                    current_alpha = new_alpha
                    if current_alpha < best_alpha:
                        best_alpha = current_alpha
                        global_best_packing = list(current_packing)
                    
        elif r < 0.9:
            # 2-swap: remove two, add two
            if len(current_packing) < 2:
                continue
            rem1, rem2 = random.sample(current_packing, 2)
            
            temp_pair_union = 0
            for idx in current_packing:
                if idx not in (rem1, rem2):
                    temp_pair_union |= tpair[idx]
            
            # Need to find two non-conflicting candidates
            candidates = get_valid_candidates(current_set - {rem1, rem2}, temp_pair_union)
            if len(candidates) >= 2:
                cand1, cand2 = random.sample(candidates, 2)
                if (tpair[cand1] & tpair[cand2]) == 0:
                    new_packing = [x for x in current_packing if x not in (rem1, rem2)] + [cand1, cand2]
                    new_alpha = compute_alpha_fast(new_packing)
                    
                    if new_alpha <= current_alpha:
                        current_packing = new_packing
                        current_set.discard(rem1)
                        current_set.discard(rem2)
                        current_set.add(cand1)
                        current_set.add(cand2)
                        current_pair_union = 0
                        for idx in current_packing:
                            current_pair_union |= tpair[idx]
                        current_alpha = new_alpha
                        if current_alpha < best_alpha:
                            best_alpha = current_alpha
                            global_best_packing = list(current_packing)
                        
        else:
            # Perturbation: Remove 1-3 random triples to escape local minima
            # Then greedily refill
            n_remove = random.randint(1, 3)
            for _ in range(n_remove):
                if current_packing:
                    rem = random.choice(current_packing)
                    current_packing.remove(rem)
                    current_set.discard(rem)
            
            # Recompute pair_union
            current_pair_union = 0
            for idx in current_packing:
                current_pair_union |= tpair[idx]
                
            # Greedily add back valid triples
            candidates = list(range(num_triples))
            random.shuffle(candidates)
            for cand in candidates:
                if cand not in current_set and (tpair[cand] & current_pair_union) == 0:
                    new_packing = current_packing + [cand]
                    new_alpha = compute_alpha_fast(new_packing)
                    if new_alpha <= current_alpha:
                        current_packing = new_packing
                        current_set.add(cand)
                        current_pair_union |= tpair[cand]
                        current_alpha = new_alpha
            
            # After perturbation and refill, update best
            if current_alpha < best_alpha:
                best_alpha = current_alpha
                global_best_packing = list(current_packing)
                
            # If alpha got worse, try to recover
            if current_alpha > best_alpha + 1:
                # Try to add back triples to improve
                candidates = get_valid_candidates(current_set, current_pair_union)
                random.shuffle(candidates)
                for cand in candidates:
                    if time.time() > deadline:
                        break
                    new_packing = current_packing + [cand]
                    new_alpha = compute_alpha_fast(new_packing)
                    if new_alpha < current_alpha:
                        current_packing = new_packing
                        current_set.add(cand)
                        current_pair_union |= tpair[cand]
                        current_alpha = new_alpha
                        if current_alpha < best_alpha:
                            best_alpha = current_alpha
                            global_best_packing = list(current_packing)
    
    # Convert best_packing (list of indices) to list of triples
    out = [list(all_triples[i]) for i in global_best_packing]
    return out

# Quick test
if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    print(f"Time: {time.time()-t0:.2f}s")
    print(f"Num triples: {len(res)}")
    # Verify linearity
    masks = set()
    for t in res:
        m = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        if m in masks:
            print("Duplicate triple found!")
        masks.add(m)
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            mi = (1 << res[i][0]) | (1 << res[i][1]) | (1 << res[i][2])
            mj = (1 << res[j][0]) | (1 << res[j][1]) | (1 << res[j][2])
            inter = mi & mj
            count = bin(inter).count('1')
            if count > 1:
                print(f"Non-linear: {res[i]} and {res[j]} share {count} points")
    
    # Compute alpha
    def check_alpha(triples):
        n = 16
        pop_arr = np.zeros(1 << n, dtype=np.int8)
        for i in range(1, 1 << n):
            pop_arr[i] = pop_arr[i >> 1] + (i & 1)
        covered = np.zeros(1 << n, dtype=bool)
        subsets = np.arange(1 << n, dtype=np.int32)
        for t in triples:
            m = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
            covered |= (subsets & m) == m
        return int(np.max(pop_arr[~covered]))
    
    print(f"Alpha: {check_alpha(res)}")
