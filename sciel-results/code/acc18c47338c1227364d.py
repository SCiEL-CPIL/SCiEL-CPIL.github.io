import time
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    N = 12
    P = 27720  # LCM(1, 2, ..., 12)
    
    def has_matching(m, L):
        """Check if (m, m+L] contains distinct b_1,...,b_12 with i|b_i."""
        # For each i in 1..12, find candidate positions in (m, m+L]
        # Candidate for i: numbers in (m, m+L] divisible by i
        # We need to assign distinct numbers to each i
        
        # Build adjacency: for each i, which "slots" (offsets) are available
        # Slot j (0-indexed) corresponds to number m+1+j
        # Edge i-j exists if (m+1+j) % i == 0
        
        # Use DFS with backtracking, ordering i by number of candidates (most constrained first)
        
        # Precompute candidates for each i
        candidates = []
        for i in range(1, N + 1):
            cand = []
            # Numbers in (m, m+L] divisible by i
            # First number > m divisible by i:
            first = ((m // i) + 1) * i
            if first > m + L:
                return False
            for x in range(first, m + L + 1, i):
                cand.append(x - m - 1)  # 0-indexed slot
            candidates.append(cand)
        
        # Sort by number of candidates (most constrained first)
        order = sorted(range(N), key=lambda i: len(candidates[i]))
        
        # Check if any i has no candidates
        for i in order:
            if len(candidates[i]) == 0:
                return False
        
        used = [False] * L
        
        def dfs(idx):
            if idx == N:
                return True
            i = order[idx]
            for slot in candidates[i]:
                if not used[slot]:
                    used[slot] = True
                    if dfs(idx + 1):
                        return True
                    used[slot] = False
            return False
        
        return dfs(0)
    
    def compute_f(m):
        """Compute f(12, m): least positive L such that (m, m+L] has the matching."""
        # f is at most 144 per the contract
        for L in range(1, 145):
            if time.time() > deadline - 5:
                return None  # timeout guard
            if has_matching(m, L):
                return L
        return 145  # shouldn't happen
    
    # We need to search over m in [0, 1000000]
    # f(12,m) depends only on m mod 27720
    # So we compute f for each residue 0..27719, find the best, then pick an m in [0,1000000] with that residue
    
    best_L = 0
    best_residue = 0
    
    # Try all 27720 residues
    for r in range(P):
        if time.time() > deadline - 10:
            break
        L = compute_f(r)
        if L is None:
            break
        if L > best_L:
            best_L = L
            best_residue = r
        # Early termination: if we found 144, that's the max possible
        if best_L >= 144:
            break
    
    # Find an m in [0, 1000000] with m % 27720 == best_residue
    # m = best_residue works if best_residue <= 1000000
    # Since best_residue < 27720 < 1000000, m = best_residue is valid
    m = best_residue
    
    # Verify it's in range
    if m < 0 or m > 1000000:
        m = 0
    
    return int(m)
