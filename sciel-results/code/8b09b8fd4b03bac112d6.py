import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    n_points = q * q
    
    # Field arithmetic F_49 = F_7[alpha], alpha^2 = 3
    # Element is (a, b) representing a + b*alpha
    # a, b in 0..6
    
    def field_add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
    
    def field_sub(a, b):
        return ((a[0] - b[0]) % p, (a[1] - b[1]) % p)
    
    def field_mul(a, b):
        # (a0 + a1*alpha) * (b0 + b1*alpha) = a0b0 + a0b1*alpha + a1b0*alpha + a1b1*alpha^2
        # = (a0b0 + 3*a1b1) + (a0b1 + a1b0)*alpha
        return ((a[0]*b[0] + 3*a[1]*b[1]) % p, (a[0]*b[1] + a[1]*b[0]) % p)
    
    def field_inv(a):
        # a = a0 + a1*alpha
        # a * (a0 - a1*alpha) = a0^2 - 3*a1^2
        # inv = (a0 - a1*alpha) / (a0^2 - 3*a1^2)
        num = (a[0], (-a[1]) % p)
        den = (a[0]*a[0] - 3*a[1]*a[1]) % p
        if den == 0:
            raise ValueError("Zero element")
        den_inv = pow(den, p-2, p)
        return field_mul(num, (den_inv, 0))
    
    # Map point index to coordinates
    # idx = xa * q + xb * p + ya * 1 ... wait
    # Let's use idx = (xa * p + xb) * q + (ya * p + yb)
    # Or simpler: idx = xa * 49 + xb * 7 + ya ... no
    # Let's use:
    # x = (xa, xb), y = (ya, yb)
    # idx = (xa * p + xb) * q + (ya * p + yb)
    
    coords = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    coords.append(((xa, xb), (ya, yb)))
                    
    # Precompute directions
    # dir[x][y] = direction index
    # Directions 0..48: slope m in F_49. m = (0,0) to (6,6)
    # Direction 49: vertical
    # m = (ya - xa) / (yb - xb) ? No.
    # Line through x, y.
    # If x_a == y_a, vertical.
    # Else slope m = (y - x) / (1) ? No.
    # Vector v = (1, m).
    # y = x + t * v.
    # y_a = x_a + t
    # y_b = x_b + t * m_b
    # t = y_a - x_a
    # m = (y_b - x_b) / (y_a - x_a)
    
    dir_matrix = np.zeros((n_points, n_points), dtype=np.int32)
    
    for i in range(n_points):
        for j in range(n_points):
            if i == j:
                dir_matrix[i][j] = -1
                continue
            xi, yi = coords[i]
            xj, yj = coords[j]
            da = (xj[0] - xi[0]) % p
            if da == 0:
                dir_matrix[i][j] = 49
            else:
                # m = (y_j - y_i) / (x_j - x_i)
                num = field_sub(yj, yi)
                den = (da, 0)
                den_inv = pow(da, p-2, p)
                m = field_mul(num, (den_inv, 0))
                # Map m to index 0..48
                # m = (m0, m1)
                # idx = m0 * p + m1
                dir_matrix[i][j] = m[0] * p + m[1]
                
    C = np.full(n_points, 50, dtype=np.int32)
    # Open directions: use a bitmask or set
    # 50 directions. Can use a 64-bit int bitmask.
    Open = [ (1 << 50) - 1 for _ in range(n_points) ]
    
    S = set()
    candidates = list(range(n_points))
    
    # To speed up, we can just iterate
    # We need to find a y such that for all x with C[x]==1, dir[x][y] is not in Open[x]
    
    # We'll maintain a list of points with C[x] == 1
    critical = []
    
    while candidates:
        # Find a valid candidate
        valid_y = None
        for y in candidates:
            is_valid = True
            for x in critical:
                d = dir_matrix[x][y]
                if d >= 0:
                    # Check if d is in Open[x]
                    if Open[x] & (1 << d):
                        is_valid = False
                        break
            if is_valid:
                valid_y = y
                break
        
        if valid_y is None:
            break
            
        # Add valid_y to S
        S.add(valid_y)
        candidates.remove(valid_y)
        
        # Update C and Open for all x != valid_y
        for x in range(n_points):
            if x == valid_y:
                continue
            d = dir_matrix[x][valid_y]
            if d >= 0 and (Open[x] & (1 << d)):
                Open[x] &= ~(1 << d)
                C[x] -= 1
                if C[x] == 1:
                    critical.append(x)
                    
    # Construct N = All points not in S
    N_indices = [i for i in range(n_points) if i not in S]
    
    # Return array of shape (k, 2, 2)
    result = np.zeros((len(N_indices), 2, 2), dtype=np.int64)
    for i, idx in enumerate(N_indices):
        x, y = coords[idx]
        result[i][0] = [x[0], x[1]]
        result[i][1] = [y[0], y[1]]
        
    return result
