import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin

    # Helper to convert matrix to list of colors
    def matrix_to_colors(mat, n):
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append(int(mat[i, j]))
        return colors

    # Build a valid 50-vertex 4-coloring using known construction
    # Use the construction from: "Four-colorings of K_n without monochromatic triangles"
    # We'll use a greedy + local search approach with a good starting point.

    # First, let's try to build a valid 50-vertex coloring using a structured approach.
    # Use the fact that the Paley graph on 49 vertices (mod 49... not prime) doesn't work directly.
    # Instead, use a random restart with aggressive local search.

    def count_monochromatic_triangles(mat, n):
        """Count total monochromatic triangles in a 4-coloring."""
        cost = 0
        for c in range(4):
            # For each vertex k, count pairs (i,j) where i<j, i!=k, j!=k, and mat[i,k]==mat[j,k]==c
            # and mat[i,j]==c
            # Faster: for each c, build adjacency and count triangles
            A = (mat == c).astype(np.int32)
            np.fill_diagonal(A, 0)
            # Number of triangles = trace(A^3) / 6
            cost += int(np.trace(A @ A @ A)) // 6
        return cost

    def local_search(mat, n, time_limit, seed=42):
        """Run local search to eliminate monochromatic triangles."""
        rng = random.Random(seed)
        cost = count_monochromatic_triangles(mat, n)
        if cost == 0:
            return mat, cost, True

        T = 50.0
        cooling = 0.9995
        best_mat = mat.copy()
        best_cost = cost

        while time.time() < time_limit:
            if cost == 0:
                return mat, cost, True

            # Pick random edge
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            if i == j:
                continue

            c_old = mat[i, j]
            c_new = rng.randint(0, 3)
            if c_new == c_old:
                continue

            # Calculate delta in cost
            # For each k != i, j:
            #   old: if mat[i,k]==c_old and mat[j,k]==c_old, triangle removed (-1)
            #   new: if mat[i,k]==c_new and mat[j,k]==c_new, triangle added (+1)
            delta = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                if mat[i, k] == c_old and mat[j, k] == c_old:
                    delta -= 1
                if mat[i, k] == c_new and mat[j, k] == c_new:
                    delta += 1

            new_cost = cost + delta
            if new_cost < 0 or rng.random() < np.exp(-delta / T):
                mat[i, j] = c_new
                mat[j, i] = c_new
                cost = new_cost
                if cost < best_cost:
                    best_cost = cost
                    best_mat = mat.copy()
            T *= cooling

        if cost < best_cost:
            best_mat = mat.copy()
            best_cost = cost

        return best_mat, best_cost, best_cost == 0

    # Strategy: Try to find a valid 50-vertex coloring
    # Use multiple random restarts with local search
    best_n = 0
    best_colors = None

    # Try n=50 with multiple restarts
    n_target = 50
    max_restarts = 20
    time_per_restart = (deadline - start_time - 5.0) / max_restarts if (deadline - start_time - 5.0) > 0 else 0

    for restart in range(max_restarts):
        if time.time() > deadline - 3.0:
            break

        # Initialize randomly
        mat = np.zeros((n_target, n_target), dtype=int)
        for i in range(n_target):
            for j in range(i + 1, n_target):
                c = random.randint(0, 3)
                mat[i, j] = c
                mat[j, i] = c

        # Run local search
        time_limit = min(deadline - 3.0, start_time + (restart + 1) * time_per_restart)
        mat, cost, success = local_search(mat, n_target, time_limit, seed=restart * 1000 + 42)

        if success:
            best_n = n_target
            best_colors = matrix_to_colors(mat, n_target)
            break

        # Keep track of best attempt
        if best_n < n_target:
            # Save the best incomplete solution for potential fallback
            pass

    # If we found n=50, try to extend to n=51
    if best_n == 50 and time.time() < deadline - 5.0:
        # Take the valid 50-vertex coloring and try to add vertex 50
        base_mat = np.zeros((50, 50), dtype=int)
        idx = 0
        for i in range(50):
            for j in range(i + 1, 50):
                base_mat[i, j] = best_colors[idx]
                base_mat[j, i] = best_colors[idx]
                idx += 1

        # Try to extend to 51
        extended = np.zeros((51, 51), dtype=int)
        extended[:50, :50] = base_mat

        # For each possible coloring of edges from new vertex 50 to vertices 0-49,
        # check if it creates any monochromatic triangles
        # A triangle (50, i, j) is monochromatic if mat[50,i] == mat[50,j] == mat[i,j]
        # So for each color c, the set of vertices connected to 50 with color c must be an independent set in color c

        found_51 = False
        for attempt in range(100):
            if time.time() > deadline - 2.0:
                break

            # Randomly assign colors from 50 to 0-49
            for i in range(50):
                c = random.randint(0, 3)
                extended[50, i] = c
                extended[i, 50] = c

            # Check for monochromatic triangles involving vertex 50
            valid = True
            for c in range(4):
                # Get vertices connected to 50 with color c
                neighbors = [i for i in range(50) if extended[50, i] == c]
                # Check if any pair in neighbors has edge color c
                for a in range(len(neighbors)):
                    for b in range(a + 1, len(neighbors)):
                        if extended[neighbors[a], neighbors[b]] == c:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break

            if valid:
                found_51 = True
                break

        if found_51:
            best_n = 51
            best_colors = matrix_to_colors(extended, 51)

    # Fallback: if we didn't find 50, try smaller n
    if best_n < 50:
        for n_try in [49, 48, 47, 46, 45, 40, 32]:
            if time.time() > deadline - 3.0:
                break

            # Try to find a valid coloring for n_try
            mat = np.zeros((n_try, n_try), dtype=int)
            for i in range(n_try):
                for j in range(i + 1, n_try):
                    c = random.randint(0, 3)
                    mat[i, j] = c
                    mat[j, i] = c

            time_limit = deadline - 3.0
            mat, cost, success = local_search(mat, n_try, time_limit, seed=999)

            if success:
                best_n = n_try
                best_colors = matrix_to_colors(mat, n_try)
                break

    # Final fallback: use a simple valid small coloring
    if best_n < 1 or best_colors is None:
        # n=4: just assign colors to avoid triangles
        # K4 with 4 colors: each edge different color works? No, need to check triangles.
        # For n=4, vertices 0,1,2,3. Edges: (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
        # Assign: (0,1)=0, (0,2)=1, (0,3)=2, (1,2)=2, (1,3)=1, (2,3)=0
        # Check triangles:
        # (0,1,2): colors 0,1,2 - not mono
        # (0,1,3): colors 0,2,1 - not mono
        # (0,2,3): colors 1,2,0 - not mono
        # (1,2,3): colors 2,1,0 - not mono
        best_n = 4
        best_colors = [0, 1, 2, 2, 1, 0]

    return {'n': best_n, 'colors': best_colors}
