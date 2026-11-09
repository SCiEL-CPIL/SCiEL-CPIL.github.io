import time
import random
import numpy as np

def _is_valid_fast(matrix, n):
    """
    Check if the coloring of K_n defined by matrix (n x n, upper triangle valid) 
    has no monochromatic triangles.
    matrix[i][j] is the color for edge (i,j).
    """
    for c in range(4):
        # Create adjacency matrix for color c
        A = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(i + 1, n):
                if matrix[i][j] == c:
                    A[i, j] = 1
                    A[j, i] = 1
        
        # Check for triangles: 
        # A triangle exists if there are u, v, w such that A[u,v]=A[v,w]=A[w,u]=1.
        # This is equivalent to: There exists an edge (u,v) in A such that 
        # (A @ A)[u,v] > 0.
        # Because (A @ A)[u,v] is the number of common neighbors of u and v.
        # If u and v are connected (A[u,v]=1) and have a common neighbor w, 
        # then u-v-w-u is a triangle.
        
        # Calculate A^2
        A2 = A @ A
        
        # Check if any edge (u,v) where A[u,v]=1 has A2[u,v] > 0
        # Note: A2 is symmetric. We can just check all entries.
        # If A[u,v] == 1 and A2[u,v] > 0, we have a triangle.
        
        # Vectorized check:
        # mask = (A == 1) & (A2 > 0)
        # if np.any(mask): return False
        
        # Optimization: A2[u,v] counts paths of length 2.
        # If A[u,v]=1, and A2[u,v] >= 1, then triangle exists.
        
        if np.any((A == 1) & (A2 > 0)):
            return False
            
    return True

def _build_gf49():
    """
    Construct GF(49) and return:
    - add_table, mul_table, inv_table for elements 0..48
    - primitive element alpha
    - log_table: ind(x) for x != 0
    """
    # Irreducible polynomial x^2 + x + 3 over GF(7)
    # Let w be a root. w^2 = -w - 3 = 6w + 4
    # Elements are a + b*w, represented as a + 7*b? No, let's use a tuple or int a + 7*b.
    # Let's use int representation: val = a + 7*b where a,b in 0..6.
    
    def norm(val):
        a = val % 7
        b = val // 7
        return a + 7*b

    def add(v1, v2):
        a1, b1 = v1 % 7, v1 // 7
        a2, b2 = v2 % 7, v2 // 7
        return (a1 + a2) % 7 + 7 * ((b1 + b2) % 7)

    def mul(v1, v2):
        a1, b1 = v1 % 7, v1 // 7
        a2, b2 = v2 % 7, v2 // 7
        # (a1 + b1*w) * (a2 + b2*w) = a1*a2 + (a1*b2 + a2*b1)*w + b1*b2*w^2
        # w^2 = 6w + 4
        # = a1*a2 + 4*b1*b2 + (a1*b2 + a2*b1 + 6*b1*b2)*w
        a = (a1*a2 + 4*b1*b2) % 7
        b = (a1*b2 + a2*b1 + 6*b1*b2) % 7
        return a + 7*b

    # Build tables
    add_table = [[add(i, j) for j in range(49)] for i in range(49)]
    mul_table = [[mul(i, j) for j in range(49)] for i in range(49)]
    
    # Inverses
    inv_table = [0] * 49
    inv_table[1] = 1
    for i in range(2, 49):
        # Find j such that mul(i, j) == 1
        for j in range(1, 49):
            if mul_table[i][j] == 1:
                inv_table[i] = j
                break
    
    # Find primitive element
    # Order of multiplicative group is 48.
    # Factors of 48: 2, 3, 4, 6, 8, 12, 16, 24, 48.
    # A generator g must have g^(48/p) != 1 for all prime factors p of 48.
    # Prime factors of 48 are 2, 3.
    # So g^24 != 1 and g^16 != 1.
    
    primitive = -1
    for g in range(2, 49):
        if mul_table[g][24] != 1 and mul_table[g][16] != 1:
            primitive = g
            break
            
    if primitive == -1:
        return None

    # Build log table
    log_table = [0] * 49
    log_table[1] = 0
    x = 1
    for k in range(1, 48):
        x = mul_table[primitive][x]
        log_table[x] = k
        
    return {
        'add': add_table,
        'mul': mul_table,
        'inv': inv_table,
        'primitive': primitive,
        'log': log_table
    }

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    # Try 49-vertex construction first
    gf49 = _build_gf49()
    if gf49 is not None:
        n = 49
        # Vertices 0..48 correspond to field elements 0..48
        # Color of edge (i, j) is determined by d = i - j (or j - i)
        # We need to map d to a color.
        # Hypothesis: Color(d) = log(d) % 4 for d != 0.
        # Check if this is valid.
        
        matrix = [[0] * n for _ in range(n)]
        valid = True
        for i in range(n):
            for j in range(i + 1, n):
                # d = i - j
                # In GF(49), subtraction is add(i, -j)
                # -j is add(0, j) inverted? No, additive inverse.
                # Since char is 7, -j = (7 - (j%7)) % 7 + 7*((7 - (j//7)) % 7)
                # Actually, add_table[i][j] gives i+j.
                # We need i - j.
                # Let's precompute neg table or just use add with negated j.
                # Neg of val a+7b is (7-a)%7 + 7*((7-b)%7)
                
                j_neg = ((7 - (j % 7)) % 7) + 7 * ((7 - (j // 7)) % 7)
                d = gf49['add'][i][j_neg]
                
                if d == 0:
                    # Should not happen for i != j
                    valid = False
                    break
                else:
                    c = gf49['log'][d] % 4
                    matrix[i][j] = c
                    matrix[j][i] = c
            if not valid:
                break
        
        if valid and _is_valid_fast(matrix, n):
            best_n = n
            best_matrix = matrix
        else:
            # If mod 4 doesn't work, try other colorings?
            # For now, fall back to 32
            best_n = 32
            best_matrix = _build_base_32()
    else:
        best_n = 32
        best_matrix = _build_base_32()

    # If we have 49, can we extend to 50?
    # Chung's 50-vertex construction is specific.
    # If we have time, we could try to extend 49 to 50.
    # But extending is hard.
    # Let's just return the best we have.
    
    # Format output
    out_colors = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            out_colors.append(best_matrix[i][j])

    return {'n': best_n, 'colors': out_colors}

def _build_base_32():
    # Base 32-vertex construction
    log_table = {}
    x = 1
    log_table[1] = 0
    for e in range(1, 15):
        x2 = 0
        for i in range(4):
            if (x >> i) & 1:
                x2 ^= (1 << (i + 1))
        if x2 & 0b10000:
            x2 ^= 0b10011
        x = x2
        log_table[x] = e
        
    n_base = 32
    matrix = [[0] * n_base for _ in range(n_base)]
    for i in range(n_base):
        for j in range(i + 1, n_base):
            if i // 16 == j // 16:
                c = log_table[i % 16 ^ j % 16] % 3
            else:
                c = 3
            matrix[i][j] = c
            matrix[j][i] = c
    return matrix
