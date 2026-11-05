import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    n = 256
    max_sum = 2 * n  # 512
    
    # Bitset for sums: sums_mask[s] is True if s is a sum of two elements in S (b <= c)
    # We use a list of bools for speed in Python, or a set of ints.
    # A set of ints is probably faster for sparse sums.
    # But checking multiples: "is any multiple of x in sums?"
    # With a set: `any(m in sums for m in range(2*x, max_sum+1, x))`
    # With a bitset (list): `any(sums_mask[m] for m in range(2*x, max_sum+1, x))`
    # List access is faster than set lookup in Python? Maybe.
    # Let's use a set of sums, it's clean.
    
    def can_add(x, sums):
        # Check if x divides any sum in sums
        # Multiples of x are 2x, 3x, ...
        # b+c > 2x is not guaranteed, but b,c > x so b+c >= 2x+2 (if distinct) or 2x (if same).
        # Actually b,c in S, and x is new min, so b,c > x.
        # Smallest sum is (x+1) + (x+1) = 2x+2? No, S contains elements > x.
        # Min element in S is > x. Let min(S) = m. m > x.
        # Min sum is 2m > 2x.
        # So we only need to check multiples of x that are >= 2*min(S).
        # But simply checking all multiples 2x, 3x... is fine.
        for m in range(2 * x, max_sum + 1, x):
            if m in sums:
                return False
        return True

    def greedy_construct(base_start):
        """
        Starts with S = [base_start, n].
        Then tries to add base_start-1, ..., 1.
        """
        S = set(range(base_start, n + 1))
        sums = set()
        
        # Pre-compute sums for the base interval
        # Optimization: for a contiguous interval [L, R], sums are [2L, 2R].
        # Are all integers in [2L, 2R] achievable?
        # Yes, for a contiguous interval of integers, the set of pair sums (b<=c) is contiguous from 2L to 2R.
        # Proof: 2L = L+L. 2L+1 = L+(L+1). ... 2R = R+R.
        # So we can just add range(2*base_start, 2*n + 1) to sums.
        
        if S:
            min_S = min(S)
            max_S = max(S)
            # Add all sums from 2*min_S to 2*max_S
            # This is valid because S is a contiguous interval.
            sums.update(range(2 * min_S, 2 * max_S + 1))
        
        # Greedily add smaller elements
        for x in range(base_start - 1, 0, -1):
            if time.time() > deadline:
                break
            if can_add(x, sums):
                S.add(x)
                # Update sums
                # New sums: 2x and x+y for y in S_old
                # Since x is the new minimum, y > x.
                sums.add(2 * x)
                for y in S:
                    if y != x:
                        sums.add(x + y)
                        
        return S

    best_solution = set()
    best_size = 0
    
    # Strategy 1: Try different base intervals
    # We know [171, 256] is valid.
    # Try bases from 150 to 180.
    for base in range(150, 181):
        if time.time() > deadline:
            break
        S_candidate = greedy_construct(base)
        if len(S_candidate) > best_size:
            best_size = len(S_candidate)
            best_solution = S_candidate
            
    # Strategy 2: Local Search / Swap
    # Try to improve best_solution
    # We will try to swap elements: remove y, add x.
    # Then try to add more.
    
    # Helper to check validity of a full set (for verification)
    def is_valid(S):
        S_list = sorted(S)
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                for k in range(j, len(S_list)):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        return False
        return True

    # Helper to try adding x to S, returning new S and sums if successful
    def try_add_to_set(S, sums, x):
        if can_add(x, sums):
            new_S = S | {x}
            new_sums = set(sums)
            new_sums.add(2 * x)
            for y in S:
                new_sums.add(x + y)
            return new_S, new_sums
        return None, None

    # Local search phase
    # We work with the best_solution found so far
    if best_solution:
        S_cur = set(best_solution)
        # Rebuild sums for S_cur
        sums_cur = set()
        S_list = sorted(S_cur)
        for i in range(len(S_list)):
            for j in range(i, len(S_list)):
                sums_cur.add(S_list[i] + S_list[j])
        
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            # Try to add any missing element
            S_cur_set = S_cur
            for x in range(1, n + 1):
                if time.time() > deadline - 5.0:
                    break
                if x not in S_cur_set:
                    new_S, new_sums = try_add_to_set(S_cur, sums_cur, x)
                    if new_S is not None:
                        S_cur = new_S
                        sums_cur = new_sums
                        improved = True
                        break # Restart loop with new set
            
            # If no direct add, try swaps
            if not improved:
                # Try swapping y in S with x not in S
                # To be efficient, we only try swaps that might unblock something.
                # But simple random swaps or systematic swaps.
                # Systematic: for each x not in S (descending), for each y in S (ascending), try swap.
                # This is O(N^2 * Cost). Might be too slow.
                # Let's do a few random restarts/swaps.
                
                # Actually, let's try a specific heuristic:
                # If we can't add x, it's because x divides some sum s.
                # s = b+c.
                # If we remove b, s might disappear.
                # But other sums involving b disappear too.
                
                # Let's just do a limited number of random swaps.
                random.seed(42)
                for _ in range(50):
                    if time.time() > deadline - 5.0:
                        break
                    if len(S_cur) < 10:
                        break
                    y = random.choice(list(S_cur))
                    S_temp = S_cur - {y}
                    # Rebuild sums for S_temp? Expensive.
                    # Just check validity of S_temp U {x} for a few x?
                    
                    # Let's just try to add x to S_temp.
                    # We need sums for S_temp.
                    S_temp_list = sorted(S_temp)
                    sums_temp = set()
                    for i in range(len(S_temp_list)):
                        for j in range(i, len(S_temp_list)):
                            sums_temp.add(S_temp_list[i] + S_temp_list[j])
                            
                    # Try to add any x not in S_temp
                    added = False
                    for x in range(n, 0, -1):
                        if x not in S_temp and x != y:
                            if can_add(x, sums_temp):
                                S_cur = S_temp | {x}
                                # Rebuild sums for S_cur
                                S_cur_list = sorted(S_cur)
                                sums_cur = set()
                                for i in range(len(S_cur_list)):
                                    for j in range(i, len(S_cur_list)):
                                        sums_cur.add(S_cur_list[i] + S_cur_list[j])
                                improved = True
                                added = True
                                break
                    if added:
                        break

    # Final verification
    if not is_valid(best_solution if not improved else S_cur):
        # Fallback to known valid set
        return list(range(171, 257))
        
    return list(best_solution if not improved else S_cur)
