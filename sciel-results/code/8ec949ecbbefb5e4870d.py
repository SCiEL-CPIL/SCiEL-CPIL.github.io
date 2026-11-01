import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix
import itertools

# EVOLVE-BLOCK-START

def _precompute_balls(q, n, radius, deadline):
    """Precompute for each word x the list of words within Hamming distance <= radius."""
    total = q ** n
    if radius == 1:
        balls = []
        powers = [1] * n
        for i in range(1, n):
            powers[i] = powers[i-1] * q
            
        for x in range(total):
            ball = [x]
            for i in range(n):
                old_digit = (x // powers[i]) % q
                for new_digit in range(q):
                    if new_digit != old_digit:
                        y = x + (new_digit - old_digit) * powers[i]
                        ball.append(y)
            balls.append(ball)
            
            if time.time() > deadline - 10:
                break
        return balls
    else:
        raise NotImplementedError("Only radius 1 is optimized")

def _greedy_cover(q, n, radius, deadline):
    total = q ** n
    balls = _precompute_balls(q, n, radius, deadline)
    
    covered = np.zeros(total, dtype=bool)
    inv = [[] for _ in range(total)]
    for i, ball in enumerate(balls):
        for y in ball:
            inv[y].append(i)
            
    counts = np.array([len(b) for b in balls], dtype=np.int32)
    selected = []
    
    while True:
        if time.time() > deadline - 5:
            break
            
        max_idx = np.argmax(counts)
        if counts[max_idx] == 0:
            break
            
        selected.append(max_idx)
        
        ball = balls[max_idx]
        for y in ball:
            if not covered[y]:
                covered[y] = True
                for i in inv[y]:
                    counts[i] -= 1
                    
        if np.all(covered):
            break

    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q
        
    codewords = []
    for idx in selected:
        digits = []
        x = idx
        for i in range(n):
            d = (x // powers[i]) % q
            digits.append(d)
        codewords.append(digits)
        
    return codewords

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    total = q ** n
    
    # Try ILP first
    try:
        # Build constraint matrix A
        # A[w, c] = 1 if c is in ball(w)
        # Note: ball(w) contains w and its neighbors.
        # So A is the adjacency matrix of the graph where edges connect words at dist <= 1.
        # Actually, A[w, c] = 1 iff d(w, c) <= radius.
        
        # We can build A row by row using the precomputed balls.
        # But _precompute_balls returns balls for each w.
        # So row w of A is 1 at indices balls[w].
        
        # Let's use a more direct construction for speed if possible, 
        # but 6561x6561 sparse is fine.
        
        # Get balls
        # We need balls for all w in 0..total-1
        # The _precompute_balls function does this.
        balls = _precompute_balls(q, n, radius, deadline)
        
        if len(balls) < total:
             raise Exception("Precomputation timed out")

        # Build sparse matrix
        rows = []
        cols = []
        data = []
        
        for w in range(total):
            for c in balls[w]:
                rows.append(w)
                cols.append(c)
                data.append(1)
        
        A = csr_matrix((data, (rows, cols)), shape=(total, total))
        
        # Objective: minimize sum(x)
        c_obj = np.ones(total)
        
        # Constraints: A @ x >= 1
        # milp uses constraints in form: lb <= A @ x <= ub
        # So lb = 1, ub = inf
        lb = np.ones(total)
        ub = np.inf * np.ones(total)
        
        # Variable bounds: 0 <= x <= 1
        res_bounds = Bounds(lb=np.zeros(total), ub=np.ones(total))
        
        # Integrality: all integer
        integrality = np.ones(total)
        
        # Solve
        # time_limit is in seconds
        remaining_time = deadline - time.time() - 5
        if remaining_time < 5:
            raise TimeoutError("Not enough time for ILP")
            
        res = milp(c=c_obj, constraints=LinearConstraint(A, lb, ub), 
                   integrality=integrality, bounds=res_bounds, 
                   options={"time_limit": remaining_time, "mip_rel_gap": 0.0})
        
        if res.status == 0: # Optimal
            x_sol = res.x
            selected_indices = np.where(x_sol > 0.5)[0]
        elif res.status == 1: # Feasible but not optimal (time limit)
            x_sol = res.x
            selected_indices = np.where(x_sol > 0.5)[0]
            # Check if it's actually a valid cover? 
            # If status is 1, it might be a partial solution. 
            # milp returns the best incumbent. If it's not a valid cover, we can't use it.
            # We must verify validity.
            # To be safe, let's verify.
            if len(selected_indices) == 0:
                 raise Exception("No solution found")
            
            # Verification
            covered = np.zeros(total, dtype=bool)
            for idx in selected_indices:
                for y in balls[idx]:
                    covered[y] = True
            if not np.all(covered):
                 raise Exception("ILP solution invalid")
        else:
            raise Exception(f"ILP failed with status {res.status}")
            
        # Convert to codewords
        powers = [1] * n
        for i in range(1, n):
            powers[i] = powers[i-1] * q
            
        codewords = []
        for idx in selected_indices:
            digits = []
            x = idx
            for i in range(n):
                d = (x // powers[i]) % q
                digits.append(d)
            codewords.append(digits)
            
        return codewords

    except Exception:
        # Fallback to greedy
        try:
            result = _greedy_cover(q, n, radius, deadline)
            return result
        except Exception:
            # Last resort fallback: direct sum construction (worse size, but valid)
            # Tetracode + full tail
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            result = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
            return result

# EVOLVE-BLOCK-END
