import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    best_set = []
    
    # We will try multiple random orderings to find a larger set
    # The greedy "smallest first" might be suboptimal.
    # However, "smallest first" tends to pack more numbers if valid.
    # Let's try "smallest first" first, then random restarts.
    
    def check_and_add(S, dp_all, dp_excl, x):
        # S is list of current elements
        # dp_all[k] is bitset of sums for subsets of S of size k
        # dp_excl[s][k] is bitset of sums for subsets of S\{s} of size k
        
        # 1. Check if x is mean of subset of S
        # Need subset T of S, |T|=k>=2, sum(T) = k*x
        for k in range(2, len(S) + 1):
            target = k * x
            if target < 0 or target > 25000: continue
            # Check if bit target is set in dp_all[k]
            if (dp_all[k] >> target) & 1:
                return False, None, None

        # 2. Check if any s in S is mean of subset of S U {x}
        # Subset must contain x.
        # T = {x} U T', T' subset S\{s}, |T'|=j>=1
        # s = (x + sum(T')) / (1+j)
        # (1+j)s = x + sum(T')
        # sum(T') = (1+j)s - x
        # Let k = 1+j. j = k-1. k>=2.
        # sum(T') = k*s - x.
        # We need subset of S\{s} of size k-1 summing to k*s - x.
        
        new_dp_all = [0] * (len(S) + 2)
        new_dp_excl = {}
        
        # We need to update dp_all and dp_excl to include x
        # But we only need to check validity.
        # If valid, we return the updated structures.
        
        # First, check validity for all s in S
        for s in S:
            # We need to check if S\{s} has subset of size j summing to V
            # V = k*s - x
            # j = k-1
            # k ranges from 2 to len(S) (since T' max size is len(S)-1, so k max len(S))
            # Wait, T' is subset of S\{s}, so max size len(S)-1.
            # So k-1 <= len(S)-1 => k <= len(S).
            for k in range(2, len(S) + 1):
                V = k * s - x
                if V < 0 or V > 25000: continue
                j = k - 1
                # Check dp_excl[s][j] for bit V
                if (dp_excl[s][j] >> V) & 1:
                    return False, None, None
        
        # If valid, update structures
        # Update dp_all
        # dp_all_new[k] = dp_all_old[k] | (dp_all_old[k-1] << x)
        for k in range(len(S) + 1, 0, -1):
            new_dp_all[k] = dp_all[k] | (dp_all[k-1] << x)
        new_dp_all[0] = 1 # Empty set sum 0
        
        # Update dp_excl
        # For each s in S, dp_excl[s] is DP for S\{s}.
        # New set is S U {x}.
        # For a fixed s in S:
        # New S\{s} = (S\{s}) U {x}.
        # So we update dp_excl[s] using x.
        for s in S:
            # dp_excl[s] currently represents S\{s}
            # We need to add x to it.
            # But we must be careful not to use the updated values in the same loop?
            # dp_excl[s] is independent for each s.
            # We can update in place if we iterate k downwards.
            for k in range(len(S), 0, -1):
                dp_excl[s][k] = dp_excl[s][k] | (dp_excl[s][k-1] << x)
            dp_excl[s][0] = 1
            
        # We also need to create dp_excl for the new element x?
        # No, dp_excl is keyed by elements in S.
        # When we add x to S, we will need dp_excl[x] for future checks.
        # dp_excl[x] should be DP for S (old).
        # So new_dp_excl[x] = copy of old dp_all.
        new_dp_excl = dict(dp_excl)
        new_dp_excl[x] = dp_all[:] # Copy old dp_all
        
        return True, new_dp_all, new_dp_excl

    # Initialize
    S = []
    dp_all = [0] * 50
    dp_all[0] = 1
    dp_excl = {}
    
    # Try greedy with smallest first
    for x in range(1, 513):
        if time.time() > deadline: break
        valid, new_dp_all, new_dp_excl = check_and_add(S, dp_all, dp_excl, x)
        if valid:
            S.append(x)
            dp_all = new_dp_all
            dp_excl = new_dp_excl
            if len(S) > len(best_set):
                best_set = S[:]
                
    # Try some random restarts if time permits
    # Randomize order
    candidates = list(range(1, 513))
    random.shuffle(candidates)
    
    for _ in range(5): # 5 restarts
        if time.time() > deadline - 10: break
        S = []
        dp_all = [0] * 50
        dp_all[0] = 1
        dp_excl = {}
        
        for x in candidates:
            if time.time() > deadline: break
            valid, new_dp_all, new_dp_excl = check_and_add(S, dp_all, dp_excl, x)
            if valid:
                S.append(x)
                dp_all = new_dp_all
                dp_excl = new_dp_excl
                if len(S) > len(best_set):
                    best_set = S[:]
                    
    return best_set

construct()
