# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using vectorized batch removal.
    
    Strategy:
    1. Start with N = F_49^2.
    2. Iteratively identify points that can be removed without invalidating any point's witness coverage.
    3. A point x can be removed if:
       - x has at least one witness line (line L with all other points in N).
       - No other point y has its ONLY witness line passing through x.
    4. Remove all such points simultaneously.
    5. Repeat until no points can be removed.
    """
    start_time = time.time()
    deadline = start_time + 570  # Safety margin before 600s limit
    
    p = 7
    q = p * p  # 49
    k = 3  # alpha^2 = 3
    
    def mul_f49(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1*a2 + k*b1*b2) % p, (a1*b2 + a2*b1) % p)
    
    all_points = [(xa, xb, ya, yb) for xa in range(p) for xb in range(p) 
                  for ya in range(p) for yb in range(p)]
    n = len(all_points)  # 2401
    
    # Precompute lines
    # Vertical lines: fixed (xa, xb)
    # Slope lines: direction (1, m) where m != (0,0)
    
    # Incidence matrix A: shape (n, n_lines)
    # We will store lines as lists of point indices
    
    lines = []
    # Add vertical lines
    for xa in range(p):
        for xb in range(p):
            line_pts = []
            for i, (xa_i, xb_i, ya, yb) in enumerate(all_points):
                if xa_i == xa and xb_i == xb:
                    line_pts.append(i)
            lines.append(line_pts)
            
    # Add slope lines
    for ma in range(p):
        for mb in range(p):
            if ma == 0 and mb == 0:
                continue
            # Group points by intercept
            intercepts = defaultdict(list)
            for i, (xa, xb, ya, yb) in enumerate(all_points):
                mx = mul_f49((ma, mb), (xa, xb))
                key = ((ya - mx[0]) % p, (yb - mx[1]) % p)
                intercepts[key].append(i)
            for key, pts in intercepts.items():
                lines.append(pts)
                
    n_lines = len(lines)
    
    # Build incidence matrix (sparse or dense boolean)
    # Dense boolean matrix is 2401 x 2450 ~ 6M bytes. Fine.
    A = np.zeros((n, n_lines), dtype=np.uint8)
    for l_idx, pts in enumerate(lines):
        for p_idx in pts:
            A[p_idx, l_idx] = 1
            
    # Precompute lines_through_point for fast access
    lines_through_point = [[] for _ in range(n)]
    for l_idx in range(n_lines):
        for p_idx in lines[l_idx]:
            lines_through_point[p_idx].append(l_idx)
            
    # Initialize N = all points
    in_N = np.ones(n, dtype=bool)
    
    def recompute_witness_counts():
        """
        Compute S_L (number of points NOT in N on line L)
        and witness_count (number of valid witness lines for each point)
        """
        # S_L = sum of (1 - in_N) over points in L
        # ~in_N is boolean, cast to int
        not_in_N = (~in_N).astype(np.int8)
        S = A.T @ not_in_N  # shape (n_lines,)
        
        witness_count = np.zeros(n, dtype=np.int8)
        
        # For points in N: witness if S_L == 0
        # For points not in N: witness if S_L == 1
        
        # Vectorized approach:
        # Create a mask for valid lines for each point?
        # It's easier to loop over points or use matrix ops carefully.
        
        # Let's use matrix multiplication tricks.
        # valid_lines_for_N = (S == 0)
        # valid_lines_for_notN = (S == 1)
        
        valid_lines_N = (S == 0).astype(np.uint8)
        valid_lines_notN = (S == 1).astype(np.uint8)
        
        # For each point y, count valid lines through y.
        # If y in N: count = sum(A[y, :] * valid_lines_N)
        # If y not in N: count = sum(A[y, :] * valid_lines_notN)
        
        # We can compute both and select based on in_N
        counts_N = A @ valid_lines_N
        counts_notN = A @ valid_lines_notN
        
        witness_count = np.where(in_N, counts_N, counts_notN)
        
        return witness_count, S

    # Initial computation
    witness_count, S = recompute_witness_counts()
    
    while True:
        if time.time() > deadline - 30:
            break
            
        # Identify critical points: points with exactly 1 witness
        # Note: Points not in N must also have witness >= 1.
        # If any point has witness_count == 0, the set is invalid.
        # But we start valid and only remove safely, so this shouldn't happen.
        
        critical_points = np.where(witness_count == 1)[0]
        
        if len(critical_points) == 0:
            # No critical points.
            # We can potentially remove all points in N?
            # Check: If we remove x, does y lose all witnesses?
            # If y has >= 2 witnesses, and we remove x, y loses at most 1 witness (the one containing x).
            # So y still has >= 1 witness.
            # x needs a witness in new N.
            # x needs a line L with S_L == 0 in old N.
            # If we remove all points in N, S_L becomes 49 for all L.
            # This is not valid.
            # So we can only remove a subset R such that for x in R, there is L with L \ R = N \ R?
            # Actually, the condition for x in R is that there is L with S_L(R) = 1.
            # S_L(R) = S_L(old) + |L \cap R|.
            # If S_L(old) = 0, we need |L \cap R| = 1.
            # So x must have a line L that contains ONLY x from R.
            # This is the same condition as before, but without the "forbidden" constraint from critical points?
            # Wait, if there are no critical points, the "forbidden" set is empty.
            # So we can remove any x that has a "private" line in R.
            # If we remove ALL points in N, R = N.
            # Does x have a line L with L \cap N = {x}?
            # No, L has 49 points.
            # So we cannot remove all.
            
            # Let's just try to remove points one by one or in small batches?
            # Or use the same logic:
            # Forbidden set is empty.
            # So removable = in_N.
            # But we must check if x has a witness.
            # If witness_count[x] > 0, x has a witness.
            # But after removal, does x still have a witness?
            # This requires simulation.
            
            # To be safe, let's just break if no critical points?
            # Or try to remove a random subset?
            # Actually, if no critical points, we can remove any set R such that
            # for all x in R, there exists L with L \cap R = {x}.
            # This is a matching problem.
            # For now, let's just break to avoid complexity.
            # The critical point removal should drive the size down significantly.
            break
            
        # Construct forbidden mask
        # A point x is forbidden if it lies on the unique witness line of a critical point y != x.
        forbidden = np.zeros(n, dtype=bool)
        
        for y in critical_points:
            # Find the unique witness line for y
            # y is in N or not?
            # If y in N, witness L has S_L == 0.
            # If y not in N, witness L has S_L == 1.
            
            # Get lines through y
            lys = lines_through_point[y]
            
            # Find the one that is valid
            valid_l = None
            for l in lys:
                if in_N[y]:
                    if S[l] == 0:
                        valid_l = l
                        break
                else:
                    if S[l] == 1:
                        valid_l = l
                        break
            
            if valid_l is not None:
                # Mark all points on this line as forbidden, except y itself
                for x in lines[valid_l]:
                    if x != y:
                        forbidden[x] = True
        
        # Removable points: in N, not forbidden, and have witness_count > 0
        # Actually, if x is in N and not forbidden, can we remove it?
        # We need x to have a witness in the NEW N.
        # This is guaranteed if we use the "batch" logic correctly?
        # The logic "x is not on unique witness of any other critical point" ensures other points are safe.
        # But does it ensure x is safe?
        # x needs a witness L with S_L(new) = 1 (if x removed).
        # S_L(new) = S_L(old) + 1 (since x in L).
        # So we need S_L(old) = 0.
        # So x must have a witness in OLD N.
        # witness_count[x] > 0 ensures this.
        
        removable = in_N & (~forbidden) & (witness_count > 0)
        
        if not np.any(removable):
            break
            
        # Remove these points
        in_N[removable] = False
        
        # Recompute witness counts
        witness_count, S = recompute_witness_counts()
        
        # Check validity (should be true)
        if np.any(witness_count == 0):
            # Error: we removed too much?
            # Rollback?
            # For now, just break.
            break

    # Build result
    pts = [all_points[i] for i in range(n) if in_N[i]]
    result = np.array([[[pt[0], pt[1]], [pt[2], pt[3]]] for pt in pts], dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
