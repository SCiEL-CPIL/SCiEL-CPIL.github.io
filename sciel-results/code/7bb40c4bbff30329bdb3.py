import time
import random

def construct():
    deadline = time.time() + 55.0
    n_max = 512
    target_size = 48
    
    def is_valid_addition(S, x):
        """
        Checks if x can be added to S without violating the condition.
        S is a list of integers.
        Returns True if valid.
        """
        n = len(S)
        if n < 2:
            return True
            
        # 1. Check if x is the mean of a subset of S
        # We need to check if there is a subset T of S with |T|=k>=2 such that sum(T) = k*x
        # We can use a DP for S.
        # To avoid recomputing DP for every x, we should maintain it.
        # But here we are in a function, so we compute it.
        # Optimization: Only compute up to max possible k.
        
        # Actually, let's maintain the DP in the outer loop.
        # This function is for the check.
        
        # For now, let's assume we have a helper that checks against a precomputed DP.
        # Since we don't have it here, we do a quick check.
        
        # Check 1: x as mean of S
        # DP for S
        dp = [0] * (n + 1)
        dp[0] = 1
        for val in S:
            for k in range(n, 0, -1):
                if dp[k-1]:
                    dp[k] |= (dp[k-1] << val)
                    
        for k in range(2, n + 1):
            target = k * x
            if target > 512 * n: # Max possible sum
                break
            if (dp[k] >> target) & 1:
                return False

        # 2. Check if any a in S becomes mean of subset of S U {x} \ {a}
        # This is expensive.
        # We can approximate or do it carefully.
        # For each a in S:
        #   Check if there is T' subset of S \ {a} such that x + sum(T') = (|T'|+1)*a
        #   sum(T') = (|T'|+1)*a - x
        
        # To make this fast, we can pre-compute DP for S \ {a} for each a?
        # That is O(N^2) memory/time.
        # Given N=48, N^2=2304. DP size is sum ~ 24k bits.
        # 2304 * 24k bits is ~ 7MB. Feasible?
        # But updating is the issue.
        
        # Let's try a simpler heuristic first: 
        # If we just run the greedy and it fails, we can do a local search.
        
        return True

    # Better approach: Maintain DP states incrementally.
    
    def greedy_build(order):
        S = []
        S_set = set()
        # dp_all[k]: bitset of sums of k elements from S
        dp_all = [1]
        
        # dp_leave[i][k]: bitset of sums of k elements from S \ {S[i]}
        # We store these as a list of lists of ints.
        dp_leave = []
        
        for x in order:
            if len(S) >= target_size:
                break
            if time.time() > deadline:
                break
            if x in S_set:
                continue
            
            valid = True
            
            # Check 1: Is x mean of subset of S?
            for k in range(2, len(S) + 1):
                if k < len(dp_all):
                    target = k * x
                    if (dp_all[k] >> target) & 1:
                        valid = False
                        break
            
            if not valid:
                continue
            
            # Check 2: Does adding x invalidate any existing a?
            # For each a_i in S (index i):
            # We need to check if a_i is mean of subset of (S \ {a_i}) U {x}
            # This subset must contain x (otherwise it was already checked in previous steps for S).
            # So subset is {x} U T, T subset of S \ {a_i}.
            # Sum(T) + x = (|T| + 1) * a_i
            # Sum(T) = (|T| + 1) * a_i - x
            
            for i, a_i in enumerate(S):
                # dp_leave[i] is DP for S \ {a_i}
                # Size of S \ {a_i} is len(S) - 1
                # T can have size j from 0 to len(S) - 1
                # But subset size is j+1. We need subset size >= 2, so j+1 >= 2 => j >= 1.
                # Also, the subset must be distinct from a_i, which is guaranteed by S \ {a_i}.
                
                # Wait, if T is empty (j=0), subset is {x}, size 1. Mean is x.
                # Condition: a_i = x. But x is not in S, so a_i != x.
                # So j=0 is not a violation (subset size 1 is not allowed by "at least two OTHER elements").
                # So we check j >= 1.
                
                for j in range(1, len(S)):
                    # We need sum(T) = (j+1)*a_i - x
                    # T has size j.
                    # Check if this sum is in dp_leave[i][j]
                    target_sum = (j + 1) * a_i - x
                    if target_sum < 0:
                        continue
                    # Max sum of j elements from 512 is j*512.
                    if target_sum > j * 512:
                        continue
                    if j < len(dp_leave[i]):
                        if (dp_leave[i][j] >> target_sum) & 1:
                            valid = False
                            break
                if not valid:
                    break
            
            if valid:
                S.append(x)
                S_set.add(x)
                
                # Update dp_all
                old_dp_all = dp_all
                new_dp_all = [0] * (len(S) + 1)
                new_dp_all[0] = 1
                for k in range(len(S) - 1, -1, -1):
                    if old_dp_all[k]:
                        new_dp_all[k] |= old_dp_all[k]
                        new_dp_all[k+1] |= (old_dp_all[k] << x)
                dp_all = new_dp_all
                
                # Update dp_leave
                new_dp_leave = []
                # For existing elements, their leave-one-out set gains x
                for i in range(len(S) - 1):
                    old_dp_leave_i = dp_leave[i]
                    # New set is (S \ {S[i]}) U {x}
                    new_dp_leave_i = [0] * (len(old_dp_leave_i) + 1)
                    new_dp_leave_i[0] = 1
                    for k in range(len(old_dp_leave_i) - 1, -1, -1):
                        if old_dp_leave_i[k]:
                            new_dp_leave_i[k] |= old_dp_leave_i[k]
                            new_dp_leave_i[k+1] |= (old_dp_leave_i[k] << x)
                    new_dp_leave.append(new_dp_leave_i)
                
                # For the new element x, its leave-one-out set is the old S
                # dp_leave_new.append(old_dp_all)
                # Note: old_dp_all has length len(S) (before adding x, len was len(S)-1? No.)
                # Before adding x, len(S) was N. dp_all had length N+1.
                # After adding x, len(S) is N+1.
                # The leave-one-out set for x is the previous S (size N).
                # So we append old_dp_all.
                new_dp_leave.append(old_dp_all)
                
                dp_leave = new_dp_leave
                
        return S

    best_set = []
    best_size = 0
    
    # Try different orders
    rng = random.Random(42)
    
    # 1. Descending
    S = greedy_build(range(n_max, 0, -1))
    if len(S) > best_size:
        best_size = len(S)
        best_set = S
        
    # 2. Ascending
    S = greedy_build(range(1, n_max + 1))
    if len(S) > best_size:
        best_size = len(S)
        best_set = S
        
    # 3. Random orders
    for i in range(50):
        if time.time() > deadline:
            break
        order = list(range(1, n_max + 1))
        rng.shuffle(order)
        S = greedy_build(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S

    # If we didn't reach 48, try local search?
    # The greedy with maintained DP is quite fast.
    # Let's see if we can just return the best found.
    
    return best_set

if __name__ == "__main__":
    s = construct()
    print(len(s))
