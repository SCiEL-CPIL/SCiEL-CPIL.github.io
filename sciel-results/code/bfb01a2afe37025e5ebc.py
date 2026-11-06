import time
import random
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    N = 1000
    best_size = 0
    best_S = set()
    
    def get_counts_from_S(S):
        """Compute all pair sums counts from set S."""
        counts = {}
        S_list = list(S)
        for i, a in enumerate(S_list):
            for b in S_list[i:]:
                s = a + b
                counts[s] = counts.get(s, 0) + 1
        return counts
    
    def is_valid_addition_fast(x, S, counts):
        """
        Check if adding x to S results in a valid set (all sums have at most 2 representations).
        Returns True if valid.
        """
        for u in S:
            s = x + u
            if counts.get(s, 0) >= 2:
                return False
        if counts.get(2 * x, 0) >= 2:
            return False
        return True
    
    def is_valid_swap(x, y, S, counts):
        """
        Check if swapping y (in S) for x (not in S) results in a valid set.
        """
        if x in S or y not in S:
            return False
        
        # Sums that will be removed (involving y)
        removed_sums = set()
        for v in S:
            if v == y:
                continue
            removed_sums.add(y + v)
        removed_sums.add(2 * y)
        
        # Check sums that will be added (involving x, excluding y since y is removed)
        for u in S:
            if u == y:
                continue
            s = x + u
            old_c = counts.get(s, 0)
            rem_c = 1 if s in removed_sums else 0
            new_c = old_c - rem_c + 1
            if new_c > 2:
                return False
        
        s = 2 * x
        old_c = counts.get(s, 0)
        rem_c = 1 if s in removed_sums else 0
        new_c = old_c - rem_c + 1
        if new_c > 2:
            return False
        
        return True
    
    def apply_addition(S, counts, x):
        """Add x to S and update counts."""
        for v in S:
            s = x + v
            counts[s] = counts.get(s, 0) + 1
        s = 2 * x
        counts[s] = counts.get(s, 0) + 1
        S.add(x)
    
    def apply_removal(S, counts, y):
        """Remove y from S and update counts."""
        for v in S:
            if v == y:
                continue
            s = y + v
            counts[s] -= 1
            if counts[s] == 0:
                del counts[s]
        s = 2 * y
        counts[s] -= 1
        if counts[s] == 0:
            del counts[s]
        S.remove(y)
    
    def apply_swap(S, counts, x, y):
        """Swap y (in S) for x (not in S)."""
        # Remove y
        for v in S:
            if v == y:
                continue
            s = y + v
            counts[s] -= 1
            if counts[s] == 0:
                del counts[s]
        s = 2 * y
        counts[s] -= 1
        if counts[s] == 0:
            del counts[s]
        S.remove(y)
        
        # Add x
        for v in S:
            s = x + v
            counts[s] = counts.get(s, 0) + 1
        s = 2 * x
        counts[s] = counts.get(s, 0) + 1
        S.add(x)
    
    def greedy_construct(p=0.15):
        """
        Probabilistic construction: randomly select each integer with probability p,
        then greedily repair by removing elements that cause sum conflicts.
        """
        S = set()
        # Randomly select candidates
        for x in range(1, N + 1):
            if random.random() < p:
                S.add(x)
        
        # Compute counts and repair
        counts = get_counts_from_S(S)
        
        # Greedy repair: repeatedly find a sum with >2 representations and remove one contributor
        max_iter = 1000
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            # Find a sum with count > 2
            bad_sum = None
            for s, c in counts.items():
                if c > 2:
                    bad_sum = s
                    break
            
            if bad_sum is None:
                break
            
            # Find all pairs in S that sum to bad_sum
            pairs = []
            S_list = list(S)
            for i, a in enumerate(S_list):
                for b in S_list[i:]:
                    if a + b == bad_sum:
                        pairs.append((a, b))
            
            if not pairs:
                # Shouldn't happen, but skip
                break
            
            # Remove one random element from a random pair
            random_pair = random.choice(pairs)
            elem_to_remove = random.choice(random_pair)
            apply_removal(S, counts, elem_to_remove)
        
        # Now try to add more elements greedily
        for x in range(1, N + 1):
            if time.time() > deadline:
                break
            if x not in S and is_valid_addition_fast(x, S, counts):
                apply_addition(S, counts, x)
        
        return S, counts
    
    def local_search(S, counts, time_budget):
        """
        Local search: try additions, swaps, and kicks to improve the set.
        """
        S = set(S)
        counts = dict(counts)
        
        while time.time() < time_budget:
            improved = False
            
            # Try adding any element
            candidates = [x for x in range(1, N + 1) if x not in S]
            random.shuffle(candidates)
            
            for x in candidates:
                if time.time() > time_budget:
                    break
                if is_valid_addition_fast(x, S, counts):
                    apply_addition(S, counts, x)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swaps
            S_list = list(S)
            random.shuffle(S_list)
            
            for y in S_list:
                if time.time() > time_budget:
                    break
                
                # Try to find an x that can replace y
                candidates = [x for x in range(1, N + 1) if x not in S]
                random.shuffle(candidates)
                
                for x in candidates:
                    if time.time() > time_budget:
                        break
                    if is_valid_swap(x, y, S, counts):
                        apply_swap(S, counts, x, y)
                        improved = True
                        break
                
                if improved:
                    break
            
            if improved:
                continue
            
            # Kick: remove 1-2 random elements, then try to add
            if len(S) > 2:
                elems = list(S)
                random.shuffle(elems)
                kick_size = random.choice([1, 2])
                to_remove = elems[:kick_size]
                
                S_orig = set(S)
                counts_orig = dict(counts)
                
                for y in to_remove:
                    apply_removal(S, counts, y)
                
                # Try to add elements
                added_any = False
                candidates = [x for x in range(1, N + 1) if x not in S]
                random.shuffle(candidates)
                
                for x in candidates:
                    if time.time() > time_budget:
                        break
                    if is_valid_addition_fast(x, S, counts):
                        apply_addition(S, counts, x)
                        added_any = True
                        # Continue trying to add more
                        # But keep the kicked elements removed for now
                        # Actually, let's try to add back the kicked elements too
                        for y in to_remove:
                            if y not in S and is_valid_addition_fast(y, S, counts):
                                apply_addition(S, counts, y)
                
                if added_any and len(S) >= len(S_orig):
                    # Keep this state
                    pass
                else:
                    # Rollback
                    S = S_orig
                    counts = counts_orig
        
        return S, counts
    
    # Main: try multiple probabilistic constructions and local searches
    best_size = 0
    best_S = set()
    
    # Try different probabilities
    for p in [0.10, 0.15, 0.20, 0.25]:
        if time.time() > deadline - 20:
            break
        
        S, counts = greedy_construct(p)
        size = len(S)
        
        if size > best_size:
            best_size = size
            best_S = set(S)
        
        # Local search with remaining time budget
        remaining = deadline - time.time()
        if remaining > 5:
            S, counts = local_search(S, counts, deadline - 5)
            size = len(S)
            if size > best_size:
                best_size = size
                best_S = set(S)
    
    # Final greedy polish on the best set
    if best_S and time.time() < deadline - 10:
        counts = get_counts_from_S(best_S)
        S = set(best_S)
        
        # Try to add more elements
        for x in range(1, N + 1):
            if time.time() > deadline - 10:
                break
            if x not in S and is_valid_addition_fast(x, S, counts):
                apply_addition(S, counts, x)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
    
    return list(best_S)

# EVOLVE-BLOCK-END
