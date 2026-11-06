import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 120.0
    safety_margin = 5.0
    
    # Build the known 32-vertex coloring as fallback
    def build_32():
        def mul(a, b):
            r = 0
            for i in range(4):
                if b >> i & 1:
                    r ^= a << i
            for i in range(7, 3, -1):
                if r >> i & 1:
                    r ^= 0b10011 << (i - 4)
            return r
        
        log, x = {}, 1
        for e in range(15):
            log[x] = e
            x = mul(x, 2)
        
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                if i // 16 == j // 16:
                    colors.append(log[i ^ j] % 3)
                else:
                    colors.append(3)
        return colors
    
    colors32 = build_32()
    best_n = 32
    best_colors = colors32
    
    # Try to build a 50-vertex coloring using GF(49) Paley-type construction
    # GF(49) = GF(7^2), represented as a + b*omega where omega^2 = 3 (irreducible over GF(7))
    # omega^2 = 3, so omega^7 = omega * (omega^2)^3 = omega * 27 = omega * 6 = 6*omega (mod 7: 27 mod 7 = 6)
    # Actually let me use the polynomial x^2 - 3 over GF(7). Check irreducibility:
    # x^2 - 3 = 0 has no solution in GF(7) since 3 is not a quadratic residue mod 7
    # (quadratic residues mod 7: 0,1,2,4)
    
    # Represent elements as (a, b) meaning a + b*omega
    # Addition: component-wise mod 7
    # Multiplication: (a,b)*(c,d) = (ac+3bd, ad+bc) mod 7
    
    def gf49_add(p, q):
        return ((p[0] + q[0]) % 7, (p[1] + q[1]) % 7)
    
    def gf49_mul(p, q):
        a, b = p
        c, d = q
        return ((a*c + 3*b*d) % 7, (a*d + b*c) % 7)
    
    def gf49_pow(p, e):
        result = (1, 0)
        base = p
        while e > 0:
            if e & 1:
                result = gf49_mul(result, base)
            base = gf49_mul(base, base)
            e >>= 1
        return result
    
    # Build all 49 elements
    elements = [(a, b) for a in range(7) for b in range(7)]
    elem_index = {elem: i for i, elem in enumerate(elements)}
    
    # Find a primitive element (generator of GF(49)*)
    # GF(49)* has order 48
    # Try (0, 1) = omega
    # omega^2 = (3, 0)
    # omega^4 = (2, 0) = 2
    # omega^7 = omega * omega^6 = omega * (omega^2)^3 = omega * 27 = omega * 6 = (0, 6)
    # Let's just find the order of (0,1)
    
    def gf49_order(p):
        if p == (0, 0):
            return 0
        result = (1, 0)
        for i in range(1, 49):
            result = gf49_mul(result, p)
            if result == (1, 0):
                return i
        return -1
    
    # Find primitive element
    primitive = None
    for elem in elements:
        if elem == (0, 0):
            continue
        if gf49_order(elem) == 48:
            primitive = elem
            break
    
    if primitive is None:
        # Fallback: just use the 32-vertex coloring
        return {'n': best_n, 'colors': best_colors}
    
    # Generate all powers of primitive
    powers = []
    current = (1, 0)
    for i in range(48):
        powers.append(current)
        current = gf49_mul(current, primitive)
    
    # Partition GF(49)* into 4 cosets of the subgroup of order 12
    # Subgroup H = {g^0, g^4, g^8, ..., g^44} where g is primitive (order 48)
    # H has order 48/4 = 12
    # Cosets: H, g*H, g^2*H, g^3*H
    
    # Color assignment: color of edge (i,j) = coset index of (i-j) in GF(49)
    # But we need to handle the case where i=j (not an edge) and i-j=0
    
    # For the 49 vertices (GF(49) elements), color edge (i,j) based on i-j
    # If i-j is in coset k, assign color k
    
    # Build color lookup: for each nonzero element, which coset it's in
    coset_membership = {}
    for k in range(4):
        for m in range(12):
            # Element g^(k + 4*m)
            elem = powers[(k + 4*m) % 48]
            coset_membership[elem] = k
    
    # Now build the 49-vertex coloring
    n49 = 49
    color_matrix = np.zeros((n49, n49), dtype=np.int8)
    
    for i in range(n49):
        for j in range(i + 1, n49):
            diff = gf49_add(elements[i], gf49_add((6, 0), gf49_add((0, 0), gf49_mul((6, 0), elements[j]))))
            # Actually: diff = elements[i] - elements[j] = elements[i] + (-1)*elements[j]
            # -1 in GF(7) is 6
            diff = gf49_add(elements[i], gf49_mul((6, 0), elements[j]))
            if diff == (0, 0):
                color_matrix[i, j] = 0
                color_matrix[j, i] = 0
            else:
                c = coset_membership[diff]
                color_matrix[i, j] = c
                color_matrix[j, i] = c
    
    # Check if this 49-vertex coloring is valid (no monochromatic triangles)
    def check_valid_49():
        for i in range(n49):
            for j in range(i + 1, n49):
                for k in range(j + 1, n49):
                    if color_matrix[i, j] == color_matrix[i, k] == color_matrix[j, k]:
                        return False
        return True
    
    if check_valid_49():
        # Great! Now try to add vertex 49 (the 50th vertex)
        # Vertex 49 connects to vertices 0..48
        # We need to assign colors to edges (49, i) for i=0..48
        # such that no monochromatic triangle is formed
        
        # For each vertex i, the color of (49, i) must be different from
        # the color of (i, j) for any j where (49, j) has the same color
        # This is a constraint satisfaction problem
        
        # Try all 4^49 possibilities? No, that's too many.
        # Use backtracking with pruning
        
        n50 = 50
        color50 = color_matrix.copy()
        color50 = np.zeros((n50, n50), dtype=np.int8)
        color50[:49, :49] = color_matrix
        color50[:49, 49] = -1  # Unassigned
        color50[49, :49] = -1
        
        def try_add_vertex_50():
            """Backtracking to assign colors to vertex 49"""
            # For vertex 49, we need to assign a color to each edge (49, i) for i=0..48
            # Constraint: for any i, j < 49, if color(49,i) == color(49,j) == color(i,j), that's a mono triangle
            
            # Use backtracking
            assignment = [-1] * 49
            
            def backtrack(idx):
                if idx == 49:
                    return True
                for c in range(4):
                    # Check if assigning color c to (49, idx) creates a mono triangle
                    valid = True
                    for j in range(idx):
                        if assignment[j] == c and color_matrix[idx, j] == c:
                            valid = False
                            break
                    if valid:
                        assignment[idx] = c
                        if backtrack(idx + 1):
                            return True
                        assignment[idx] = -1
                return False
            
            if backtrack(0):
                for i in range(49):
                    color50[49, i] = assignment[i]
                    color50[i, 49] = assignment[i]
                return True
            return False
        
        if try_add_vertex_50():
            # Verify the 50-vertex coloring
            def check_valid_50():
                for i in range(n50):
                    for j in range(i + 1, n50):
                        for k in range(j + 1, n50):
                            if color50[i, j] == color50[i, k] == color50[j, k]:
                                return False
                return True
            
            if check_valid_50():
                # Convert to flat list in lexicographic order
                colors50 = []
                for i in range(n50):
                    for j in range(i + 1, n50):
                        colors50.append(int(color50[i, j]))
                return {'n': n50, 'colors': colors50}
    
    # If 50-vertex construction failed, try local search to improve
    if time.time() - start_time < deadline - safety_margin:
        # Use the 49-vertex coloring as a starting point if valid, else 32-vertex
        if check_valid_49():
            n_start = 49
            start_matrix = color_matrix.copy()
        else:
            # Build 32-vertex matrix
            n_start = 32
            start_matrix = np.zeros((n_start, n_start), dtype=np.int8)
            for idx, c in enumerate(colors32):
                # Find (i,j) from idx
                i = 0
                remaining = idx
                while i < n_start:
                    count_in_row = n_start - 1 - i
                    if remaining < count_in_row:
                        j = i + 1 + remaining
                        break
                    remaining -= count_in_row
                    i += 1
                start_matrix[i, j] = c
                start_matrix[j, i] = c
        
        # Local search with numpy for speed
        n_target = 50
        if n_start < n_target:
            # Extend matrix
            full_matrix = np.zeros((n_target, n_target), dtype=np.int8)
            full_matrix[:n_start, :n_start] = start_matrix
            # Initialize new edges randomly
            for i in range(n_start, n_target):
                for j in range(i + 1, n_target):
                    full_matrix[i, j] = random.randint(0, 3)
                    full_matrix[j, i] = full_matrix[i, j]
            for i in range(n_start):
                for j in range(n_start, n_target):
                    full_matrix[i, j] = random.randint(0, 3)
                    full_matrix[j, i] = full_matrix[i, j]
        else:
            full_matrix = start_matrix.copy()
        
        # Efficient local search using numpy
        def count_mono_triangles(matrix, n):
            """Count monochromatic triangles using matrix operations"""
            count = 0
            for c in range(4):
                # Adjacency matrix for color c
                adj = (matrix == c).astype(np.int8)
                np.fill_diagonal(adj, 0)
                # Number of triangles = trace(adj^3) / 6
                # But this counts each triangle 6 times (3! orientations)
                # Actually trace(A^3) counts each triangle 6 times for undirected
                # For a triangle (i,j,k): A[i,j]*A[j,k]*A[k,i] contributes to trace
                # trace(A^3) = sum_i (A^3)[i,i] = sum_{i,j,k} A[i,j]*A[j,k]*A[k,i]
                # Each triangle contributes 6 (all 6 cyclic orderings)
                # So number of triangles = trace(A^3) / 6
                # But we only want triangles where all 3 edges have color c
                # Since adj only has 1s for color c edges, this is correct
                a2 = adj @ adj
                a3 = a2 @ adj
                count += int(np.trace(a3)) // 6
            return count
        
        # Local search
        best_count = count_mono_triangles(full_matrix, n_target)
        max_iter = 20000
        no_improve = 0
        
        for it in range(max_iter):
            if time.time() - start_time > deadline - safety_margin:
                break
            
            # Pick random edge
            i = random.randint(0, n_target - 1)
            j = random.randint(0, n_target - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            old_color = int(full_matrix[i, j])
            new_color = random.randint(0, 3)
            if old_color == new_color:
                continue
            
            # Compute delta in triangle count
            # Changing edge (i,j) from old_color to new_color:
            # - Breaks old_color triangles containing (i,j): count = number of k with matrix[i,k]==old_color and matrix[j,k]==old_color
            # - Creates new_color triangles containing (i,j): count = number of k with matrix[i,k]==new_color and matrix[j,k]==new_color
            
            row_i = full_matrix[i]
            row_j = full_matrix[j]
            
            # For old_color: count k where both i-k and j-k are old_color
            # Exclude k=i and k=j
            mask_i = (row_i == old_color).astype(np.int8)
            mask_j = (row_j == old_color).astype(np.int8)
            mask_i[i] = 0
            mask_i[j] = 0
            mask_j[i] = 0
            mask_j[j] = 0
            old_tri = int(np.sum(mask_i * mask_j))
            
            mask_i2 = (row_i == new_color).astype(np.int8)
            mask_j2 = (row_j == new_color).astype(np.int8)
            mask_i2[i] = 0
            mask_i2[j] = 0
            mask_j2[i] = 0
            mask_j2[j] = 0
            new_tri = int(np.sum(mask_i2 * mask_j2))
            
            delta = new_tri - old_tri
            
            if delta < 0:
                full_matrix[i, j] = new_color
                full_matrix[j, i] = new_color
                best_count += delta
                no_improve = 0
            elif delta == 0 and random.random() < 0.05:
                full_matrix[i, j] = new_color
                full_matrix[j, i] = new_color
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > 500:
                    # Restart with some perturbation
                    for _ in range(50):
                        ri = random.randint(0, n_target - 1)
                        rj = random.randint(0, n_target - 1)
                        if ri != rj:
                            nc = random.randint(0, 3)
                            full_matrix[ri, rj] = nc
                            full_matrix[rj, ri] = nc
                    best_count = count_mono_triangles(full_matrix, n_target)
                    no_improve = 0
        
        # Final check
        final_count = count_mono_triangles(full_matrix, n_target)
        if final_count == 0:
            colors_final = []
            for i in range(n_target):
                for j in range(i + 1, n_target):
                    colors_final.append(int(full_matrix[i, j]))
            return {'n': n_target, 'colors': colors_final}
        
        # If not valid, try to salvage by reducing n
        # Find the largest valid prefix
        for n_try in range(n_target, best_n, -1):
            sub_matrix = full_matrix[:n_try, :n_try]
            if count_mono_triangles(sub_matrix, n_try) == 0:
                colors_final = []
                for i in range(n_try):
                    for j in range(i + 1, n_try):
                        colors_final.append(int(sub_matrix[i, j]))
                return {'n': n_try, 'colors': colors_final}
    
    return {'n': best_n, 'colors': best_colors}
