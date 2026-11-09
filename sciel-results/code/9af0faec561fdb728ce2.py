<<<<<<< SEARCH
import numpy as np

def construct():
    N = 256
    S = set()
    candidates = list(range(1, N + 1))
    
    # Helper to check if adding x to S creates a 4-AP
    def creates_4ap(x, S):
        # x is 1st: x, x+d, x+2d, x+3d
        for d in range(1, (N - x) // 3 + 1):
            if (x + d) in S and (x + 2 * d) in S and (x + 3 * d) in S:
                return True
        # x is 2nd: x-d, x, x+d, x+2d
        for d in range(1, min(x, (N - x) // 2) + 1):
            if (x - d) in S and (x + d) in S and (x + 2 * d) in S:
                return True
        # x is 3rd: x-2d, x-d, x, x+d
        for d in range(1, min(x // 2, N - x) + 1):
            if (x - 2 * d) in S and (x - d) in S and (x + d) in S:
                return True
        # x is 4th: x-3d, x-2d, x-d, x
        for d in range(1, x // 3 + 1):
            if (x - 3 * d) in S and (x - 2 * d) in S and (x - d) in S:
                return True
        return False

    # Helper to calculate gain in 3-APs if x is added to S
    def calc_gain(x, S):
        gain = 0
        # x is 1st: x, x+d, x+2d
        for d in range(1, (N - x) // 2 + 1):
            if (x + d) in S and (x + 2 * d) in S:
                gain += 1
        # x is 2nd: x-d, x, x+d
        for d in range(1, min(x, N - x) + 1):
            if (x - d) in S and (x + d) in S:
                gain += 1
        # x is 3rd: x-2d, x-d, x
        for d in range(1, x // 2 + 1):
            if (x - 2 * d) in S and (x - d) in S:
                gain += 1
        return gain

    while candidates:
        best_x = None
        best_gain = -1
        
        for x in candidates:
            if creates_4ap(x, S):
                continue
            
            gain = calc_gain(x, S)
            
            if gain > best_gain:
                best_gain = gain
                best_x = x
            elif gain == best_gain:
                # Tie-breaker: prefer smaller x? Or maybe larger x to spread out?
                # Smaller x is fine.
                if best_x is None or x < best_x:
                    best_x = x
        
        if best_x is None:
            break
            
        S.add(best_x)
        candidates.remove(best_x)
        
    return sorted(S)
=======
import numpy as np

def construct():
    N = 256
    in_S = np.zeros(N + 1, dtype=bool)
    candidates = set(range(1, N + 1))
    S = set()
    
    while candidates:
        # Calculate gain and validity for all candidates
        gains = np.zeros(N + 1, dtype=int)
        valid = np.ones(N + 1, dtype=bool)
        
        A = in_S[1:N+1]
        
        max_d = N // 3
        
        for d in range(1, max_d + 1):
            # 3-AP Gains
            if N >= 2*d + 1:
                # 1. x, x+d, x+2d
                gains[1:N-2*d+1] += (A[d:N-d] * A[2*d:N]).astype(int)
                # 2. x-d, x, x+d
                gains[d+1:N-d+1] += (A[0:N-2*d] * A[2*d:N]).astype(int)
                # 3. x-2d, x-d, x
                gains[2*d+1:N+1] += (A[0:N-2*d] * A[d:N-d]).astype(int)

            # 4-AP Violations
            if N >= 3*d + 1:
                # 1. x, x+d, x+2d, x+3d
                viol_mask = (A[d:N-2*d] & A[2*d:N-d] & A[3*d:N])
                if np.any(viol_mask):
                    valid[1:N-3*d+1] &= ~viol_mask
                    
                # 2. x-d, x, x+d, x+2d
                if N-2*d >= d+1:
                    viol_mask = (A[0:N-3*d] & A[2*d:N-d] & A[3*d:N])
                    if np.any(viol_mask):
                        valid[d+1:N-2*d+1] &= ~viol_mask

                # 3. x-2d, x-d, x, x+d
                if N-d >= 2*d+1:
                    viol_mask = (A[0:N-3*d] & A[d:N-2*d] & A[3*d:N])
                    if np.any(viol_mask):
                        valid[2*d+1:N-d+1] &= ~viol_mask

                # 4. x-3d, x-2d, x-d, x
                if N >= 4*d + 1:
                    viol_mask = (A[0:N-4*d] & A[d:N-3*d] & A[2*d:N-2*d])
                    if np.any(viol_mask):
                        valid[3*d+1:N+1] &= ~viol_mask
                # If N < 4d+1, no x can satisfy x-3d >= 1 and x <= N with 4 elements fitting?
                # x=3d+1 requires 4d+1 <= N. So if N < 4d+1, this case yields no violations.

        best_x = -1
        best_gain = -1
        
        for x in candidates:
            if valid[x]:
                g = gains[x]
                if g > best_gain:
                    best_gain = g
                    best_x = x
                elif g == best_gain:
                    if best_x == -1 or x < best_x:
                        best_x = x
                        
        if best_x == -1:
            break
            
        S.add(best_x)
        candidates.remove(best_x)
        in_S[best_x] = True
        
    return sorted(S)
>>>>>>> REPLACE
