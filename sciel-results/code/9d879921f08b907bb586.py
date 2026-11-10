import numpy as np
import time

def nikodym_set() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580
    
    p = 7
    k = 3  # alpha^2 = 3 in F_49 = F_7[alpha]
    
    # Field operations
    def mul(a, b):
        a1, a2 = a
        b1, b2 = b
        return ((a1*b1 + k*a2*b2) % p, (a1*b2 + a2*b1) % p)
    
    # All points as (xa, xb, ya, yb)
    all_points = [(xa, xb, ya, yb) for xa in range(p) for xb in range(p)
                  for ya in range(p) for yb in range(p)]
    n = len(all_points)  # 2401
    
    # Hermitian unital: Tr(x) + Norm(y) = 0
    # Tr((xa,xb)) = xa + xb (conjugate of xb is xb since we're in F_p extension... 
    # Actually for F_{p^2}/F_p, Tr(a+b*alpha) = (a+b*alpha) + (a+b*alpha)^p = (a+b*alpha) + (a-b*alpha) = 2a
    # Norm((ya,yb)) = (ya+yb*alpha)(ya-yb*alpha) = ya^2 - yb^2*alpha^2 = ya^2 - k*yb^2
    
    def in_unital(xa, xb, ya, yb):
        # Tr(x) = 2*xa mod p (since conjugate of xb*alpha is -xb*alpha for p odd, but actually in F_{p^2}, 
        # the nontrivial automorphism sends alpha -> alpha^p = alpha^{7} = alpha^{6}*alpha = alpha^{-1}*alpha = 1? No.
        # For F_49 = F_7[alpha]/(alpha^2-3), alpha^7 = alpha (Frobenius). Since alpha^2 = 3, alpha^4 = 9 = 2, 
        # alpha^6 = 6, alpha^7 = alpha*6 = 6alpha. So conjugate of alpha is 6alpha = -alpha.
        # So Tr(a+b*alpha) = (a+b*alpha) + (a+6b*alpha) = 2a + 7b*alpha = 2a.
        # Norm(a+b*alpha) = (a+b*alpha)(a+6b*alpha) = a^2 + 7ab*alpha + 6b^2*alpha^2 = a^2 + 18b^2 = a^2 + 4b^2
        # Wait, let me recompute: (a+bα)(a+6bα) = a² + 6abα + abα + 6b²α² = a² + 7abα + 6b²·3 = a² + 0 + 18b² = a² + 4b²
        # But 4b² = 4b² mod 7. And k=3, so -k = -3 = 4 mod 7. So Norm = ya² - k·yb² = ya² - 3yb².
        # Check: ya² - 3yb² vs ya² + 4yb². Since -3 ≡ 4 mod 7, yes they match.
        return (2 * xa + ya * ya - k * yb * yb) % p == 0
    
    # Find all unital points
    unital_points = []
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        if in_unital(xa, xb, ya, yb):
            unital_points.append(i)
    
    # The unital has q+1 = 50 points
    # Strategy: exclude unital points where ya is a quadratic residue mod p
    # QR = {0, 1, 2, 4} (since 0²=0, 1²=1, 2²=4, 3²=2, 4²=2, 5²=4, 6²=1)
    qr = set((u * u) % p for u in range(p))
    
    # S = unital points where ya is NOT in qr (to maximize S, we want to exclude as many as possible)
    # But we need to verify this gives a valid Nikodym set.
    # Alternative: try both and pick the larger valid one.
    
    def build_S(use_qr):
        """Build S as unital points where ya (if use_qr) is in qr, else not in qr."""
        S = set()
        for idx in unital_points:
            xa, xb, ya, yb = all_points[idx]
            if use_qr and ya in qr:
                S.add(idx)
            elif not use_qr and ya not in qr:
                S.add(idx)
        return S
    
    def is_valid_S(in_S):
        """Check if S is a valid excluded set (N = complement is Nikodym)."""
        # Precompute lines
        # For each direction (49 slopes + vertical), group points into lines
        direction_lines = []
        
        # Vertical lines: same (xa, xb)
        vert_dict = {}
        for i, (xa, xb, ya, yb) in enumerate(all_points):
            key = (xa, xb)
            if key not in vert_dict:
                vert_dict[key] = []
            vert_dict[key].append(i)
        direction_lines.append(list(vert_dict.values()))
        
        # Slope m = (ma, mb)
        for ma in range(p):
            for mb in range(p):
                lines_dict = {}
                for i, (xa, xb, ya, yb) in enumerate(all_points):
                    mx = mul((xa, xb), (ma, mb))
                    c = ((ya - mx[0]) % p, (yb - mx[1]) % p)
                    if c not in lines_dict:
                        lines_dict[c] = []
                    lines_dict[c].append(i)
                direction_lines.append(list(lines_dict.values()))
        
        num_dirs = len(direction_lines)
        
        # For each point, check if there's a direction where all other points on the line are in N
        for x in range(n):
            has_safe = False
            for d in range(num_dirs):
                # Find which line in direction d contains x
                for line in direction_lines[d]:
                    if x in line:
                        clean = True
                        for q in line:
                            if q != x and q in in_S:
                                clean = False
                                break
                        if clean:
                            has_safe = True
                        break
            if not has_safe:
                return False
        return True
    
    # Try both variants
    best_S = None
    best_size = 0
    
    for use_qr in [True, False]:
        S = build_S(use_qr)
        size = len(S)
        if size > best_size:
            # Validate
            if is_valid_S(S):
                best_size = size
                best_S = S
    
    # If neither works, fall back to empty S (N = full plane, always valid)
    if best_S is None:
        best_S = set()
    
    # Build result: N = All \ S
    pts = [all_points[i] for i in range(n) if i not in best_S]
    result = np.array([[[pt[0], pt[1]], [pt[2], pt[3]]] for pt in pts], dtype=np.int64)
    return result
