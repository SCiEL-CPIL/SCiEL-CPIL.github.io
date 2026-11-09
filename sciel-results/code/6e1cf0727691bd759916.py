import time
import random
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    def _time_left():
        return deadline - time.time()

    def _verify(n, colors):
        """
        Check if the coloring is valid (no C4 in any color class).
        Uses bitmasks for efficiency.
        """
        if len(colors) != n * (n - 1) // 2:
            return False
        if any(c < 0 or c > 4 for c in colors):
            return False
            
        # adj[c][i] is a bitmask of neighbors of i in color c
        adj = [[0] * n for _ in range(5)]
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        
        for c in range(5):
            for u in range(n):
                mask_u = adj[c][u]
                for v in range(u + 1, n):
                    common = mask_u & adj[c][v]
                    if common & (common - 1):
                        return False
        return True

    def _cyclic_construction(n):
        """
        Cyclic construction for n=11 (distances 1..5).
        """
        if n % 2 == 0:
            return None
        k = n // 2
        if k > 5:
            return None
        
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = j - i
                dist = min(d, n - d)
                c = dist - 1
                colors.append(c)
        return colors

    def _product_construction():
        """
        Lazebnik-Woldar construction for n=26.
        Vertices are (x, y, z) in Z_3 x Z_3 x Z_3.
        """
        n = 26
        verts = list(product(range(3), repeat=3))
        colors = []
        for idx1, (a, b, c) in enumerate(verts):
            for idx2, (a2, b2, c2) in enumerate(verts):
                if idx1 >= idx2:
                    continue
                dx = (a - a2) % 2
                dy = (b - b2) % 2
                dz = (c - c2) % 2
                colors.append(dx + 2 * dy + 4 * dz)
        
        return {'n': n, 'colors': colors}

    def _local_search(n, time_budget):
        """
        Simulated annealing / hill climbing with C4 penalty.
        """
        if _time_left() < 2.0:
            return None

        num_edges = n * (n - 1) // 2
        # Initialize with random colors
        colors = [random.randint(0, 4) for _ in range(num_edges)]
        
        # Precompute edge indices to (u, v)
        edge_map = []
        for i in range(n):
            for j in range(i + 1, n):
                edge_map.append((i, j))
        
        # Build adj masks for initial state
        adj = [[0] * n for _ in range(5)]
        def build_adj(colors):
            for c in range(5):
                for i in range(n):
                    adj[c][i] = 0
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    c = colors[idx]
                    adj[c][i] |= (1 << j)
                    adj[c][j] |= (1 << i)
                    idx += 1
        
        build_adj(colors)
        
        def get_badness():
            bad = 0
            for c in range(5):
                for u in range(n):
                    for v in range(u + 1, n):
                        common = adj[c][u] & adj[c][v]
                        if common & (common - 1):
                            cnt = bin(common).count("1")
                            bad += cnt * (cnt - 1) // 2
            return bad

        best_badness = get_badness()
        best_colors = colors[:]
        
        # Simulated Annealing parameters
        T = 1.0
        cooling_rate = 0.995
        min_T = 0.01
        
        max_steps = 20000
        no_improve = 0
        max_no_improve = 5000
        
        for step in range(max_steps):
            if _time_left() < 2.0:
                break
            
            if best_badness == 0:
                return best_colors
            
            if no_improve > max_no_improve:
                # Restart with random or keep best
                break
            
            # Pick random edge
            idx = random.randint(0, num_edges - 1)
            u, v = edge_map[idx]
            
            old_c = colors[idx]
            new_c = random.randint(0, 4)
            if new_c == old_c:
                continue
            
            # Update adj
            adj[old_c][u] &= ~(1 << v)
            adj[old_c][v] &= ~(1 << u)
            adj[new_c][u] |= (1 << v)
            adj[new_c][v] |= (1 << u)
            
            new_badness = get_badness()
            
            delta = new_badness - best_badness
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                colors[idx] = new_c
                best_badness = new_badness
                if new_badness < best_badness:
                    best_colors = colors[:]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Revert
                adj[new_c][u] &= ~(1 << v)
                adj[new_c][v] &= ~(1 << u)
                adj[old_c][u] |= (1 << v)
                adj[old_c][v] |= (1 << u)
                no_improve += 1
            
            T *= cooling_rate
            if T < min_T:
                T = min_T

        if best_badness == 0:
            return best_colors
        return None

    def _spectral_relaxation(n, time_budget):
        """
        Spectral relaxation approach to guide discrete search.
        This is a structural change from pure local search.
        We minimize trace(A_c^2) for each color class c, which counts 2 * (number of P3s).
        Minimizing P3s tends to make graphs more regular and sparse, reducing C4 potential.
        """
        if _time_left() < 2.0:
            return None

        num_edges = n * (n - 1) // 2
        # Initialize with random colors
        colors = [random.randint(0, 4) for _ in range(num_edges)]
        
        # Precompute edge indices to (u, v)
        edge_map = []
        for i in range(n):
            for j in range(i + 1, n):
                edge_map.append((i, j))
        
        # Build adjacency matrices for each color
        # A_c is an n x n numpy array
        A = [np.zeros((n, n), dtype=np.float64) for _ in range(5)]
        
        def build_A(colors):
            for c in range(5):
                A[c] = np.zeros((n, n), dtype=np.float64)
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    c = colors[idx]
                    A[c][i, j] = 1.0
                    A[c][j, i] = 1.0
                    idx += 1
        
        build_A(colors)
        
        def get_spectral_penalty():
            # Sum of trace(A_c^2) for all c
            # trace(A_c^2) = sum(A_c^2) = 2 * (number of edges) + sum of degrees? No.
            # trace(A^2) = sum_i (A^2)_ii = sum_i sum_j A_ij A_ji = sum_{i,j} A_ij^2 = 2 * |E|
            # Wait, trace(A^2) is constant for a fixed graph.
            # The proxy for C4 is actually related to the number of 4-cycles.
            # Number of C4s is proportional to trace(A^4).
            # So we should minimize trace(A_c^4).
            penalty = 0.0
            for c in range(5):
                A2 = A[c] @ A[c]
                A4 = A2 @ A2
                penalty += np.trace(A4)
            return penalty

        # Gradient descent on the "soft" assignment?
        # This is complex. Let's stick to a guided local search where the move acceptance
        # is based on the change in trace(A^4) for the affected color classes.
        
        best_penalty = get_spectral_penalty()
        best_colors = colors[:]
        
        T = 10.0
        cooling_rate = 0.99
        min_T = 0.1
        
        max_steps = 10000
        no_improve = 0
        max_no_improve = 2000
        
        for step in range(max_steps):
            if _time_left() < 2.0:
                break
            
            if no_improve > max_no_improve:
                break
            
            idx = random.randint(0, num_edges - 1)
            u, v = edge_map[idx]
            
            old_c = colors[idx]
            new_c = random.randint(0, 4)
            if new_c == old_c:
                continue
            
            # Update A
            A[old_c][u, v] = 0.0
            A[old_c][v, u] = 0.0
            A[new_c][u, v] = 1.0
            A[new_c][v, u] = 1.0
            
            new_penalty = get_spectral_penalty()
            
            delta = new_penalty - best_penalty
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                colors[idx] = new_c
                if new_penalty < best_penalty:
                    best_penalty = new_penalty
                    best_colors = colors[:]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Revert
                A[new_c][u, v] = 0.0
                A[new_c][v, u] = 0.0
                A[old_c][u, v] = 1.0
                A[old_c][v, u] = 1.0
                no_improve += 1
            
            T *= cooling_rate
            if T < min_T:
                T = min_T

        if best_penalty == 0:
            # Check if valid
            if _verify(n, best_colors):
                return best_colors
        return None

    # 1. Try known algebraic constructions
    best_n = 11
    best_colors = _cyclic_construction(11)
    if best_colors and _verify(11, best_colors):
        pass # Already valid
    
    # Try n=26 construction
    if _time_left() > 5.0:
        res26 = _product_construction()
        if _verify(res26['n'], res26['colors']):
            best_n = 26
            best_colors = res26['colors']
    
    # Try to find n=27 or 28 via spectral relaxation / local search
    # This is the new structural attempt.
    if best_n < 28:
        # Try n=27 first, as it's the next step.
        for n in [27, 28]:
            if _time_left() < 15.0:
                break
            
            found = False
            # Try spectral relaxation first
            for attempt in range(3):
                if _time_left() < 5.0:
                    break
                res = _spectral_relaxation(n, 5.0)
                if res:
                    if _verify(n, res):
                        best_n = n
                        best_colors = res
                        found = True
                        break
            
            # If spectral failed, try standard local search
            if not found:
                for attempt in range(3):
                    if _time_left() < 5.0:
                        break
                    res = _local_search(n, 5.0)
                    if res:
                        if _verify(n, res):
                            best_n = n
                            best_colors = res
                            found = True
                            break
            
            if found:
                # Found a valid coloring for n, try next?
                # If we found 27, we can try 28 if time permits.
                pass
            else:
                # If we failed 27, 28 is likely harder, but let's try if time permits?
                # Given the difficulty, breaking might be safer to return 26.
                # But the directive says "Spare time does not establish a bottleneck... Keep that claim conditional".
                # Let's break to ensure we return a valid result quickly.
                break

    # Final verification
    if best_colors is None or not _verify(best_n, best_colors):
        # Fallback to 11
        best_n = 11
        best_colors = _cyclic_construction(11)

    return {'n': best_n, 'colors': best_colors}

if __name__ == "__main__":
    res = construct()
    print(res['n'])
