import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = 140.0  # Safety margin

    def seed_solution():
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    if n != 8 or q != 3 or radius != 1:
        return seed_solution()

    try:
        # 1. Generate Candidates: Shortened Ternary Golay Code
        # Generator matrix for [12, 6, 6] ternary Golay code
        G = np.array([
            [1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            [0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2],
            [0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 2, 1],
            [0, 0, 0, 1, 0, 0, 1, 1, 1, 2, 1, 1],
            [0, 0, 0, 0, 1, 0, 1, 1, 2, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 2, 1, 1, 1, 1]
        ], dtype=np.int64)

        coeffs = np.array(list(itertools.product(range(3), repeat=6)), dtype=np.int64)
        codewords_12 = (coeffs @ G) % 3
        candidates = codewords_12[:, 4:]  # Shorten to length 8
        n_candidates = len(candidates)
        
        # 2. Precompute Encodings and Balls
        # Encode points as base-3 integers
        def encode_point(p):
            val = 0
            for d in p:
                val = val * 3 + d
            return val

        # Precompute ball indices for each candidate
        # ball_indices[i] is a list of 17 point indices covered by candidate i
        ball_indices = []
        for i in range(n_candidates):
            cw = candidates[i]
            idx = encode_point(cw)
            balls = [idx]
            for pos in range(n):
                orig = cw[pos]
                for nv in range(3):
                    if nv != orig:
                        nb = cw.copy()
                        nb[pos] = nv
                        balls.append(encode_point(nb))
            ball_indices.append(np.array(balls, dtype=np.int32))

        total_points = 3 ** n
        best_solution = None
        best_size = float('inf')

        # 3. Greedy Set Cover with Randomized Tie-breaking
        # We run this a few times to find a good initial solution
        num_restarts = 5
        for r in range(num_restarts):
            if time.time() > deadline - 20:
                break
                
            # Randomize candidate order slightly for diversity
            indices = list(range(n_candidates))
            random.shuffle(indices)
            
            covered = np.zeros(total_points, dtype=np.int8)
            selected = []
            remaining = set(indices)
            
            # Greedy selection
            while remaining:
                if time.time() > deadline - 20:
                    break
                
                best_idx = None
                best_gain = -1
                
                # Optimization: Instead of checking all remaining, we can just iterate.
                # With 729 candidates, this is fast.
                for idx in remaining:
                    ca = ball_indices[idx]
                    gain = np.sum(covered[ca] == 0)
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = idx
                    elif gain == best_gain and best_idx is not None:
                        # Tie-breaking: random or by index
                        if random.random() < 0.5:
                            best_idx = idx
                
                if best_idx is None or best_gain == 0:
                    break
                
                selected.append(best_idx)
                covered[ball_indices[best_idx]] = 1
                remaining.remove(best_idx)
                
                if np.sum(covered) == total_points:
                    break
            
            if len(selected) < best_size:
                # Verify coverage just in case
                if np.sum(covered) == total_points:
                    best_size = len(selected)
                    best_solution = selected[:]

        if best_solution is None:
            return seed_solution()

        # 4. Minimal Reduction (Local Search)
        # Try to remove codewords from the greedy solution
        # Sort by "uniqueness" or just iterate
        current_solution = best_solution[:]
        current_covered = np.zeros(total_points, dtype=np.int8)
        for idx in current_solution:
            current_covered[ball_indices[idx]] = 1
        
        # We maintain a count of how many selected codewords cover each point
        # This allows O(1) check for validity upon removal
        cover_count = np.zeros(total_points, dtype=np.int16)
        for idx in current_solution:
            cover_count[ball_indices[idx]] += 1

        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 5:
                break
            
            # Shuffle to avoid bias
            order = current_solution[:]
            random.shuffle(order)
            
            for idx in order:
                if time.time() > deadline - 5:
                    break
                
                # Tentatively remove
                balls = ball_indices[idx]
                for p in balls:
                    cover_count[p] -= 1
                
                # Check if any point is now uncovered
                if np.any(cover_count == 0):
                    # Invalid removal, restore
                    for p in balls:
                        cover_count[p] += 1
                else:
                    # Valid removal
                    current_solution.remove(idx)
                    improved = True
        
        if len(current_solution) < best_size:
            best_size = len(current_solution)
            best_solution = current_solution[:]

        # 5. Optional: 1-Exchange Local Search
        # Try to swap one selected with one unselected to enable further reduction?
        # This is complex. Given the time, Greedy+Reduction is usually sufficient to hit 486.
        
        result = [list(candidates[i]) for i in best_solution]
        
        # Final validation
        if len(result) == 0:
            return seed_solution()
            
        return result

    except Exception:
        return seed_solution()
