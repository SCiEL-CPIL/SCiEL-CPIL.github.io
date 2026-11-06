import random
import time
import sys

N = 256
MAX_SUM = 2 * N

def construct():
    deadline = time.time() + 55  # Safety margin before 60s kill
    
    def is_sidon_mask(S_mask):
        """Check if S_mask represents a Sidon set."""
        # Reconstruct sums from mask is O(N^2), but we maintain Sum_mask incrementally.
        # This function is for final verification only.
        S = [i for i in range(N+1) if (S_mask >> i) & 1]
        sums = set()
        for i in range(len(S)):
            a = S[i]
            for j in range(i, len(S)):
                b = S[j]
                s = a + b
                if s in sums:
                    return False
                sums.add(s)
        return True

    def is_maximal(S_mask, Sum_mask):
        """Check if S_mask is inclusion-maximal given its Sum_mask."""
        S_list = [i for i in range(N+1) if (S_mask >> i) & 1]
        S_set = set(S_list)
        
        for x in range(1, N + 1):
            if (S_mask >> x) & 1:
                continue
            # Check if x can be added
            # 1. 2x not in Sum_mask
            if (Sum_mask >> (2 * x)) & 1:
                continue # Blocked by diagonal
            # 2. a+x not in Sum_mask for all a in S
            blocked = False
            # Iterate through set bits of S_mask
            m = S_mask
            while m:
                a = (m & -m).bit_length() - 1
                m &= m - 1
                if (Sum_mask >> (a + x)) & 1:
                    blocked = True
                    break
            if not blocked:
                # If not blocked by existing sums, x can be added
                # (We don't need to check collisions among new sums a+x as they are distinct)
                return False
        return True

    def greedy_with_blocking(rng, time_budget):
        """
        Greedy construction: add element that blocks most others.
        Returns (S_mask, Sum_mask, S_list)
        """
        start = time.time()
        S_mask = 0
        Sum_mask = 0
        S_list = []
        
        while True:
            if time.time() - start > time_budget:
                break
            
            # Check if already maximal
            if is_maximal(S_mask, Sum_mask):
                break
            
            # Find candidates that can be added
            candidates = []
            for x in range(1, N + 1):
                if (S_mask >> x) & 1:
                    continue
                # Check if x can be added
                if (Sum_mask >> (2 * x)) & 1:
                    continue
                blocked = False
                m = S_mask
                while m:
                    a = (m & -m).bit_length() - 1
                    m &= m - 1
                    if (Sum_mask >> (a + x)) & 1:
                        blocked = True
                        break
                if not blocked:
                    candidates.append(x)
            
            if not candidates:
                break # Should not happen if not maximal, but safety
            
            # Evaluate blocking power of each candidate
            best_x = None
            best_block_count = -1
            
            # Optimization: If many candidates, sample?
            # For N=256, candidates <= 256. Evaluation is O(N * |S|).
            # |S| grows. 256 * 256 * 20 ~ 1.3M ops. Might be slow in Python.
            # Let's sample if candidates > 50.
            if len(candidates) > 50:
                candidates = rng.sample(candidates, 50)
            
            for x in candidates:
                # Simulate adding x
                new_S_mask = S_mask | (1 << x)
                # New sums: {a+x for a in S} U {2x}
                
                # Count how many y in [1, N] \ (S U {x}) are blocked by new state
                blocked_count = 0
                # We need to check if y can be added to S U {x}
                # y is blocked if:
                # 1. 2y in New_Sum_mask
                # 2. a+y in New_Sum_mask for some a in S U {x}
                
                # Precompute New_Sum_mask bits? Or just check on fly?
                # Let's construct a boolean array or just check conditions.
                
                # Condition for y to be addable to S' = S U {x}:
                # - 2y not in Sum(S')
                # - a+y not in Sum(S') for all a in S'
                
                # Sum(S') = Sum(S) U {a+x for a in S} U {2x}
                
                # Iterate y
                # To speed up, we can just check the conditions.
                # This inner loop is O(N * |S'|).
                # Total O(Candidates * N * |S|).
                # 50 * 256 * 20 = 256,000 ops. Acceptable.
                
                for y in range(1, N + 1):
                    if (new_S_mask >> y) & 1:
                        continue
                    
                    # Check if y is blocked in S'
                    # 1. Check 2y against Sum(S')
                    if (Sum_mask >> (2 * y)) & 1:
                        blocked_count += 1
                        continue
                    # Check if 2y is in new sums: {a+x} or {2x}
                    # 2y == a+x ?
                    # 2y == 2x ? (y==x, but y!=x here)
                    
                    # We can just check if 2y is in the set of new sums.
                    # New sums are specific values.
                    # Instead of building the mask, check if 2y is in {a+x} or 2x.
                    # 2y == 2x is false.
                    # 2y == a+x => a = 2y - x.
                    # Check if (2y - x) is in S.
                    if (2 * y - x) >= 1 and (2 * y - x) <= N and (S_mask >> (2 * y - x)) & 1:
                        blocked_count += 1
                        continue
                    
                    # 2. Check a+y against Sum(S') for a in S'
                    # S' = S U {x}
                    # Check a in S:
                    m = S_mask
                    found_collision = False
                    while m:
                        a = (m & -m).bit_length() - 1
                        m &= m - 1
                        s = a + y
                        # Is s in Sum(S')?
                        # s in Sum(S)?
                        if (Sum_mask >> s) & 1:
                            found_collision = True
                            break
                        # s in {b+x for b in S}?
                        # s = b + x => b = s - x = a + y - x
                        # Check if (a + y - x) is in S
                        b_val = a + y - x
                        if 1 <= b_val <= N and (S_mask >> b_val) & 1:
                            found_collision = True
                            break
                        # s in {2x}?
                        if s == 2 * x:
                            found_collision = True
                            break
                    if found_collision:
                        blocked_count += 1
                        continue
                        
                    # Check a=x:
                    # s = x + y
                    # Is x+y in Sum(S')?
                    # 1. In Sum(S)?
                    if (Sum_mask >> (x + y)) & 1:
                        blocked_count += 1
                        continue
                    # 2. In {b+x for b in S}?
                    # x+y = b+x => b=y. y not in S. So no.
                    # 3. In {2x}?
                    if x + y == 2 * x: # y=x, impossible
                        blocked_count += 1
                        continue
                        
                    # If we reach here, y is NOT blocked.
                    # So it's addable.
            
            # Pick best
            # We need to re-evaluate or store best.
            # The loop above didn't store best properly. Let's refactor.
            
            best_x = None
            best_block_count = -1
            for x in candidates:
                new_S_mask = S_mask | (1 << x)
                blocked_count = 0
                for y in range(1, N + 1):
                    if (new_S_mask >> y) & 1:
                        continue
                    # Check if y is blocked
                    # 1. 2y in Sum(S')?
                    # Sum(S') = Sum(S) U {a+x | a in S} U {2x}
                    if (Sum_mask >> (2 * y)) & 1:
                        blocked_count += 1
                        continue
                    # Check if 2y == a+x
                    # a = 2y - x
                    if 1 <= 2 * y - x <= N and (S_mask >> (2 * y - x)) & 1:
                        blocked_count += 1
                        continue
                    # Check if 2y == 2x (y=x, no)
                    
                    # 2. a+y in Sum(S') for some a in S'
                    # a in S:
                    blocked = False
                    m = S_mask
                    while m:
                        a = (m & -m).bit_length() - 1
                        m &= m - 1
                        s = a + y
                        if (Sum_mask >> s) & 1:
                            blocked = True
                            break
                        b_val = a + y - x
                        if 1 <= b_val <= N and (S_mask >> b_val) & 1:
                            blocked = True
                            break
                        if s == 2 * x:
                            blocked = True
                            break
                    if blocked:
                        blocked_count += 1
                        continue
                        
                    # a = x:
                    s = x + y
                    if (Sum_mask >> s) & 1:
                        blocked_count += 1
                        continue
                    # x+y = b+x => b=y (not in S)
                    # x+y = 2x => y=x (no)
                    
                    # If not blocked, y is addable.
            
            if blocked_count > best_block_count:
                best_block_count = blocked_count
                best_x = x
            
            if best_x is None:
                break
            
            # Add best_x
            S_list.append(best_x)
            S_mask |= (1 << best_x)
            # Update Sum_mask
            # New sums: {a + best_x for a in S_list[:-1]} U {2 * best_x}
            for a in S_list[:-1]:
                Sum_mask |= (1 << (a + best_x))
            Sum_mask |= (1 << (2 * best_x))
            
            # Early stop if small enough?
            if len(S_list) >= 8:
                 if is_maximal(S_mask, Sum_mask):
                     break
                     
        return S_mask, Sum_mask, S_list

    def local_search(S_mask, Sum_mask, S_list, rng, time_budget):
        start = time.time()
        best_size = len(S_list)
        best_S_mask = S_mask
        best_Sum_mask = Sum_mask
        best_S_list = S_list
        
        while time.time() - start < time_budget:
            improved = False
            # Try removing each element
            for i in range(len(best_S_list)):
                if time.time() - start > time_budget:
                    break
                
                x_remove = best_S_list[i]
                # Construct new state without x_remove
                # This is O(N^2) if done naively.
                # Reconstructing Sum_mask from scratch is O(K^2). K~10. 100 ops.
                # Reconstructing S_mask is O(N).
                
                new_S_list = [x for x in best_S_list if x != x_remove]
                new_S_mask = 0
                for x in new_S_list:
                    new_S_mask |= (1 << x)
                
                new_Sum_mask = 0
                for i2 in range(len(new_S_list)):
                    a = new_S_list[i2]
                    for j2 in range(i2, len(new_S_list)):
                        b = new_S_list[j2]
                        new_Sum_mask |= (1 << (a + b))
                
                # Check if maximal
                if is_maximal(new_S_mask, new_Sum_mask):
                    # Success! Reduced size
                    best_S_list = new_S_list
                    best_S_mask = new_S_mask
                    best_Sum_mask = new_Sum_mask
                    best_size = len(best_S_list)
                    improved = True
                    break # Restart search with new smaller set
            
            if not improved:
                # Try swap?
                # Remove x, add y.
                # If we can swap such that we can *then* remove z, we gain.
                # This is complex. Let's just do random restarts if no improvement.
                # Or simple swap: Remove x, add y (same size).
                # If the new set allows a removal, we gain.
                
                # To keep it simple and fast, we rely on multiple greedy runs.
                # If we get stuck, break.
                break
                
        return best_S_mask, best_Sum_mask, best_S_list

    best_global_size = N + 1
    best_global_mask = 0
    best_global_sum = 0
    best_global_list = []
    
    # Phase 1: Multiple Greedy Runs
    for seed in range(100):
        if time.time() > deadline - 5:
            break
        
        rng = random.Random(seed)
        S_mask, Sum_mask, S_list = greedy_with_blocking(rng, time_budget=2.0)
        
        if not S_list:
            continue
            
        # Verify
        if not is_sidon_mask(S_mask):
            continue
        if not is_maximal(S_mask, Sum_mask):
            continue
            
        if len(S_list) < best_global_size:
            best_global_size = len(S_list)
            best_global_mask = S_mask
            best_global_sum = Sum_mask
            best_global_list = S_list
            
        if best_global_size <= 8:
            break

    # Phase 2: Local Search on Best
    if best_global_list and time.time() < deadline - 2:
        rng = random.Random(999)
        S_mask, Sum_mask, S_list = local_search(best_global_mask, best_global_sum, best_global_list, rng, time_budget=min(10, deadline - time.time() - 1))
        
        if is_sidon_mask(S_mask) and is_maximal(S_mask, Sum_mask):
            if len(S_list) < best_global_size:
                best_global_size = len(S_list)
                best_global_mask = S_mask
                best_global_sum = Sum_mask
                best_global_list = S_list

    if not best_global_list:
        # Fallback: Deterministic Greedy
        S_mask = 0
        Sum_mask = 0
        S_list = []
        for x in range(1, N + 1):
            if (S_mask >> x) & 1:
                continue
            if (Sum_mask >> (2 * x)) & 1:
                continue
            blocked = False
            m = S_mask
            while m:
                a = (m & -m).bit_length() - 1
                m &= m - 1
                if (Sum_mask >> (a + x)) & 1:
                    blocked = True
                    break
            if not blocked:
                S_list.append(x)
                S_mask |= (1 << x)
                for a in S_list[:-1]:
                    Sum_mask |= (1 << (a + x))
                Sum_mask |= (1 << (2 * x))
        return S_list

    return best_global_list

if __name__ == "__main__":
    construct()
