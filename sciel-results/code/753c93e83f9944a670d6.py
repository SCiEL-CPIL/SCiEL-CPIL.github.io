import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # LCM of 1 to 24
    L = 1
    for i in range(1, 25):
        L = (L * i) // math.gcd(L, i)
        
    # Weights W[i] = L / i
    W = [0] * 25
    for i in range(1, 25):
        W[i] = L // i
        
    def is_subset_sum(target, candidates):
        """
        Returns True if target can be formed by a sum of a subset of candidates.
        candidates is a list of integers.
        """
        if target == 0:
            return True
        if not candidates:
            return False
        
        # Sort descending for better pruning
        cands = sorted(candidates, reverse=True)
        
        # Quick check: if sum of all < target, return False
        if sum(cands) < target:
            return False
        
        # Memoization for (index, remaining_target) might help but state space is large.
        # Simple DFS with pruning is usually fast enough for N<=24.
        
        def dfs(idx, rem):
            if rem == 0:
                return True
            if idx == len(cands):
                return False
            if rem < 0:
                return False
            
            # Pruning: if sum of remaining elements < rem, fail
            # Precompute suffix sums for cands
            # This is done outside or we can just use a simple bound.
            # Let's precompute suffix sums.
            
            # Try including cands[idx]
            if dfs(idx + 1, rem - cands[idx]):
                return True
            
            # Try excluding cands[idx]
            # Pruning: if rem > sum(cands[idx+1:]), then we MUST include cands[idx]
            # But we already tried including. If that failed, and rem > suffix_sum[idx+1], then fail.
            if rem > suffix_sums[idx + 1]:
                return False
                
            if dfs(idx + 1, rem):
                return True
                
            return False
            
        # Precompute suffix sums
        suffix_sums = [0] * (len(cands) + 1)
        for i in range(len(cands) - 1, -1, -1):
            suffix_sums[i] = suffix_sums[i + 1] + cands[i]
            
        # If total sum < target, already handled.
        if suffix_sums[0] < target:
            return False
            
        return dfs(0, target)

    S = []
    W_S = []
    
    for i in range(1, 25):
        if time.time() > deadline:
            break
            
        new_W = W[i]
        # Candidates for checking new element: W_S
        # Candidates for checking old elements: W_S (excluding the specific one) + new_W
        
        # 1. Check if new_W is a subset sum of W_S
        if W_S and is_subset_sum(new_W, W_S):
            continue
        
        # 2. Check if any w in W_S is a subset sum of (W_S \ {w}) U {new_W}
        # This is equivalent to checking if w - new_W is a subset sum of (W_S \ {w})
        # AND w > new_W (since all weights are positive)
        
        valid = True
        for j in range(len(W_S)):
            w = W_S[j]
            if w <= new_W:
                continue
            # Check if w - new_W is a subset sum of W_S excluding w
            # Note: w - new_W > 0
            candidates = [W_S[k] for k in range(len(W_S)) if k != j]
            if is_subset_sum(w - new_W, candidates):
                valid = False
                break
        
        if valid:
            S.append(i)
            W_S.append(new_W)
            
    return S

# Test locally? No, just output.
