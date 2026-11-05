import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def get_cost(A):
        """Calculate sum of squares of C6 counts for each color."""
        cost = 0.0
        for c in range(4):
            Ac = A[c]
            # A2 = Ac @ Ac
            # A4 = A2 @ A2
            # A6 = A4 @ A2
            # trace(A6) is 12 * (# C6)
            A2 = Ac @ Ac
            A4 = A2 @ A2
            A6 = A4 @ A2
            c6_count = np.trace(A6) / 12.0
            cost += c6_count * c6_count
        return cost

    def is_valid(A):
        """Check if no color class contains a C6."""
        for c in range(4):
            Ac = A[c]
            A2 = Ac @ Ac
            A4 = A2 @ A2
            A6 = A4 @ A2
            if np.trace(A6) > 0:
                return False
        return True

    def extract_colors(A, n):
        """Extract color list in lexicographic order."""
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                for c in range(4):
                    if A[c][i][j] == 1:
                        colors.append(c)
                        break
        return colors

    def simulated_annealing(n, time_limit):
        """Run SA for a specific n and time limit."""
        # Initialize random coloring
        A = [np.zeros((n, n), dtype=np.int8) for _ in range(4)]
        for i in range(n):
            for j in range(i + 1, n):
                c = random.randint(0, 3)
                A[c][i][j] = 1
                A[c][j][i] = 1
        
        current_cost = get_cost(A)
        if current_cost == 0:
            return A
        
        best_A = [A[c].copy() for c in range(4)]
        best_cost = current_cost
        
        temp = 5.0  # Start with higher temp to escape local minima
        cooling = 0.9995
        start = time.time()
        
        # Pre-calculate list of edges for faster random access?
        # random.randint is fine.
        
        while time.time() - start < time_limit:
            # Pick random edge
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            
            # Find current color
            old_c = -1
            for c in range(4):
                if A[c][i][j] == 1:
                    old_c = c
                    break
            
            # Pick new color
            new_c = random.randint(0, 3)
            if new_c == old_c:
                continue
            
            # Apply move
            A[old_c][i][j] = 0
            A[old_c][j][i] = 0
            A[new_c][i][j] = 1
            A[new_c][j][i] = 1
            
            new_cost = get_cost(A)
            
            # Acceptance criterion
            if new_cost <= current_cost:
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_A = [A[c].copy() for c in range(4)]
                    if best_cost == 0:
                        return best_A
            else:
                # Boltzmann probability
                # Avoid overflow in exp
                delta = new_cost - current_cost
                if delta / temp < 700: 
                    prob = np.exp(-delta / temp)
                else:
                    prob = 0.0
                
                if random.random() < prob:
                    current_cost = new_cost
                else:
                    # Revert
                    A[new_c][i][j] = 0
                    A[new_c][j][i] = 0
                    A[old_c][i][j] = 1
                    A[old_c][j][i] = 1
            
            temp *= cooling
            
        return best_A

    # Allocate time to n values
    # Prioritize larger n
    time_allocations = {
        17: 60.0,
        16: 20.0,
        15: 10.0,
        14: 5.0,
        13: 3.0,
        12: 2.0,
        11: 1.0,
        10: 1.0,
        9: 0.5
    }
    
    best_result = None
    best_n_found = 0
    
    for n in range(17, 9, -1):
        if time.time() > deadline - 2.0:
            break
            
        alloc_time = time_allocations.get(n, 1.0)
        # Don't exceed remaining time
        remaining = deadline - time.time() - 1.0
        if remaining <= 0:
            break
        
        current_alloc = min(alloc_time, remaining)
        
        # Run SA restarts until time runs out for this n
        while time.time() < deadline - 2.0:
            # If we have less than 1s left, stop
            if deadline - time.time() < 1.0:
                break
                
            A = simulated_annealing(n, current_alloc)
            
            if is_valid(A):
                colors = extract_colors(A, n)
                best_result = {'n': n, 'colors': colors}
                best_n_found = n
                break # Found valid for this n, move to next lower n? 
                       # Actually, we want MAX n. So if we found n=17, we are done.
                       # If we found n=16, we should have tried 17 first.
                       # Since we iterate 17 down, the first success is the max.
                return best_result
        
        # If we didn't find a valid one for this n, continue to next lower n
        # But wait, if we fail for 17, we try 16.
        # If we find 16, we return 16.
        # This is correct.
        
    # Fallback if nothing found (should not happen for n<=9)
    if best_result is None:
        # Try n=9 specifically with a bit more time if available?
        # Or just return a known valid small graph
        n = 9
        colors = [0] * (n * (n - 1) // 2) # All 0 is invalid for C6? 
        # K9 has C6. We need a valid one.
        # Cyclic coloring: color(i,j) = (j-i) % 4 ?
        # Let's just return n=9 with a random valid one if we can find one quickly, 
        # or a hardcoded one if we had one. 
        # The seed program returned n=9.
        # Let's try to find n=9 quickly.
        if time.time() < deadline - 1.0:
             A = simulated_annealing(9, 1.0)
             if is_valid(A):
                 colors = extract_colors(A, 9)
                 return {'n': 9, 'colors': colors}
        
        # Hardcoded fallback for n=9 (Cyclic Z9 coloring might work? No, Z9 has C6 in some classes?)
        # Let's just return n=9 with a dummy valid structure if we must.
        # Actually, the seed program had a fallback.
        # I will use a simple heuristic for n=9.
        n = 9
        A = [np.zeros((n, n), dtype=np.int8) for _ in range(4)]
        for i in range(n):
            for j in range(i + 1, n):
                c = (j - i) % 4
                A[c][i][j] = 1
                A[c][j][i] = 1
        
        # Verify
        if is_valid(A):
            colors = extract_colors(A, n)
            return {'n': n, 'colors': colors}
        else:
            # Last resort: return n=1 (always valid)
            return {'n': 1, 'colors': []}

