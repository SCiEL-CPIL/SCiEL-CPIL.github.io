"""Construct a 4-regular graph with girth 10 on 320 vertices (the (4,10)-cage).

Uses a voltage graph lift of K_{4,4} over Z_40 with carefully chosen voltages.
The base K_{4,4} has 8 vertices and 16 edges. The lift over Z_40 gives 320 vertices.
The girth of the lift is at least 10 if the voltage assignment avoids short cycles.
"""
import time
import sys
import random
from collections import deque
from itertools import product

def construct(k, g):
    """Construct a k-regular graph with girth at least g on as few vertices as possible."""
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill
    
    if k != 4 or g != 10:
        # Fallback for other parameters: use LUW construction
        return _construct_luw(k, g, deadline)
    
    # For k=4, g=10: construct the 320-vertex cage via voltage graph lift of K_{4,4} over Z_40
    # K_{4,4} has vertices A = {a0..a3} and B = {b0..b3}, all edges a_i-b_j.
    # Lift over Z_40: 8*40 = 320 vertices.
    # Vertex (side, i, k) where side=0 for A, side=1 for B, i in 0..3, k in 0..39.
    # Edge from (0, i, k) to (1, j, (k + volt[i][j]) % 40).
    
    MOD = 40
    N_A = 4
    N_B = 4
    
    def _build_lift(voltages):
        """Build the voltage graph lift of K_{4,4} over Z_40."""
        n = 320
        edges = []
        for i in range(4):
            for j in range(4):
                volt = voltages[i][j]
                for k in range(40):
                    a_idx = i * 40 + k
                    b_idx = 160 + j * 40 + (k + volt) % 40
                    edges.append((a_idx, b_idx))
        return n, edges
    
    # Precompute all closed-walk voltage-sum expressions.
    # A closed walk in K_{4,4} alternates A->B->A->B->...->A.
    # For a closed walk of length 2m: a0->b1->a1->b2->a2->...->b_m->a0
    # Voltage sum = v[a0][b1] - v[a1][b1] + v[a1][b2] - v[a2][b2] + ... + v[a_{m-1}][b_m] - v[a0][b_m]
    # Each term is ±v[ai][bj]. We represent as a dict { (i,j): coeff } where coeff is +1 or -1.
    
    def _enumerate_closed_walks(max_length):
        """Enumerate all closed walks in K_{4,4} up to max_length (even lengths only).
        Returns list of (walk, voltage_expr) where voltage_expr is a list of (i,j,sign) tuples.
        """
        walks = []
        for length in range(4, max_length + 1, 2):
            # Closed walk: a0 -> b1 -> a1 -> b2 -> a2 -> ... -> b_{length/2} -> a0
            # a0 is fixed (we'll iterate over it), b1..b_{length/2} and a1..a_{length/2-1} vary
            half = length // 2
            for a0 in range(4):
                # We need to choose b1..b_half and a1..a_{half-1}
                # The walk is: a0, b1, a1, b2, a2, ..., b_{half-1}, a_{half-1}, b_half, a0
                # Voltage: v[a0][b1] - v[a1][b1] + v[a1][b2] - v[a2][b2] + ... + v[a_{half-1}][b_half] - v[a0][b_half]
                
                # Generate all combinations of b1..b_half and a1..a_{half-1}
                if half == 2:
                    # length=4: a0->b1->a1->b2->a0
                    # Voltage: v[a0][b1] - v[a1][b1] + v[a1][b2] - v[a0][b2]
                    for b1 in range(4):
                        for a1 in range(4):
                            for b2 in range(4):
                                expr = [
                                    (a0, b1, +1),
                                    (a1, b1, -1),
                                    (a1, b2, +1),
                                    (a0, b2, -1),
                                ]
                                walks.append(expr)
                elif half == 3:
                    # length=6: a0->b1->a1->b2->a2->b3->a0
                    # Voltage: v[a0][b1] - v[a1][b1] + v[a1][b2] - v[a2][b2] + v[a2][b3] - v[a0][b3]
                    for b1 in range(4):
                        for a1 in range(4):
                            for b2 in range(4):
                                for a2 in range(4):
                                    for b3 in range(4):
                                        expr = [
                                            (a0, b1, +1),
                                            (a1, b1, -1),
                                            (a1, b2, +1),
                                            (a2, b2, -1),
                                            (a2, b3, +1),
                                            (a0, b3, -1),
                                        ]
                                        walks.append(expr)
                elif half == 4:
                    # length=8: a0->b1->a1->b2->a2->b3->a3->b4->a0
                    # Voltage: v[a0][b1] - v[a1][b1] + v[a1][b2] - v[a2][b2] + v[a2][b3] - v[a3][b3] + v[a3][b4] - v[a0][b4]
                    for b1 in range(4):
                        for a1 in range(4):
                            for b2 in range(4):
                                for a2 in range(4):
                                    for b3 in range(4):
                                        for a3 in range(4):
                                            for b4 in range(4):
                                                expr = [
                                                    (a0, b1, +1),
                                                    (a1, b1, -1),
                                                    (a1, b2, +1),
                                                    (a2, b2, -1),
                                                    (a2, b3, +1),
                                                    (a3, b3, -1),
                                                    (a3, b4, +1),
                                                    (a0, b4, -1),
                                                ]
                                                walks.append(expr)
        return walks
    
    # Precompute closed walks for lengths 4, 6, 8
    all_walks = _enumerate_closed_walks(8)
    
    # For each walk, compute the linear form: sum of coeff * v[i][j]
    # We'll represent voltages as a flat array of 16 values: v[i][j] -> index i*4+j
    def _eval_walk(expr, volt_flat):
        """Evaluate the voltage sum for a given walk expression and voltage assignment."""
        total = 0
        for (i, j, sign) in expr:
            total += sign * volt_flat[i * 4 + j]
        return total % MOD
    
    def _check_all_walks(volt_flat):
        """Check that no closed walk of length 4, 6, or 8 has voltage sum 0 mod 40."""
        for expr in all_walks:
            if _eval_walk(expr, volt_flat) == 0:
                return False
        return True
    
    def _count_violations(volt_flat):
        """Count how many closed walks have voltage sum 0 mod 40."""
        count = 0
        for expr in all_walks:
            if _eval_walk(expr, volt_flat) == 0:
                count += 1
        return count
    
    def _verify_girth_exact(n, edges, g, deadline):
        """Verify girth >= g using non-backtracking walks from every vertex."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        max_len = g - 1  # we check walks of length 1..g-1
        
        for start_v in range(n):
            if time.time() > deadline:
                return False
            
            # states = set of (v, p) pairs reachable by non-backtracking walk of current length
            states = set()
            for w in adj[start_v]:
                states.add((w, start_v))
            
            for d in range(2, max_len + 1):
                new_states = set()
                for (v, p) in states:
                    for w in adj[v]:
                        if w != p:
                            new_states.add((w, v))
                if any(v == start_v for (v, p) in new_states):
                    return False
                states = new_states
                if not states:
                    break
        
        return True
    
    # Strategy 1: Structured parametric search
    # volt[i][j] = (a*i + b*j + c*i*j + d) mod 40
    candidate_volt_flat = None
    
    # Search over parameters a, b, c, d
    # c should be odd to avoid degeneracy
    for c in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39]:
        for d in range(0, 40, 1):
            for a in range(0, 40, 1):
                for b in range(0, 40, 1):
                    volt_flat = [(a*i + b*j + c*i*j + d) % MOD for i in range(4) for j in range(4)]
                    if _check_all_walks(volt_flat):
                        candidate_volt_flat = volt_flat
                        break
                if candidate_volt_flat:
                    break
            if candidate_volt_flat:
                break
        if candidate_volt_flat:
            break
    
    # Strategy 2: Random search with local repair
    if candidate_volt_flat is None:
        random.seed(42)
        best_volt = None
        best_violations = float('inf')
        
        for attempt in range(50000):
            if time.time() > deadline - 60:
                break
            
            volt_flat = [random.randint(0, 39) for _ in range(16)]
            violations = _count_violations(volt_flat)
            
            if violations == 0:
                candidate_volt_flat = volt_flat
                break
            
            if violations < best_violations:
                best_violations = violations
                best_volt = volt_flat[:]
            
            # Local repair: try to fix individual voltages
            for _ in range(10):
                if time.time() > deadline - 60:
                    break
                # Pick a random voltage to change
                idx = random.randint(0, 15)
                old_val = volt_flat[idx]
                # Try all other values
                best_new_val = None
                best_new_violations = violations
                for new_val in range(40):
                    if new_val == old_val:
                        continue
                    volt_flat[idx] = new_val
                    v = _count_violations(volt_flat)
                    if v < best_new_violations:
                        best_new_violations = v
                        best_new_val = new_val
                    if v == 0:
                        break
                if best_new_val is not None:
                    volt_flat[idx] = best_new_val
                    violations = best_new_violations
                    if violations == 0:
                        candidate_volt_flat = volt_flat
                        break
                else:
                    volt_flat[idx] = old_val
        
        if candidate_volt_flat is None and best_volt is not None and best_violations <= 5:
            # Try more aggressive repair from the best solution
            volt_flat = best_volt[:]
            violations = best_violations
            for _ in range(100):
                if time.time() > deadline - 60:
                    break
                if violations == 0:
                    candidate_volt_flat = volt_flat
                    break
                # Try changing each voltage to minimize violations
                improved = False
                for idx in range(16):
                    old_val = volt_flat[idx]
                    best_new_val = None
                    best_new_violations = violations
                    for new_val in range(40):
                        if new_val == old_val:
                            continue
                        volt_flat[idx] = new_val
                        v = _count_violations(volt_flat)
                        if v < best_new_violations:
                            best_new_violations = v
                            best_new_val = new_val
                        if v == 0:
                            break
                    if best_new_val is not None:
                        volt_flat[idx] = best_new_val
                        violations = best_new_violations
                        improved = True
                        if violations == 0:
                            candidate_volt_flat = volt_flat
                            break
                    else:
                        volt_flat[idx] = old_val
                    if violations == 0:
                        break
                if not improved:
                    # Perturb randomly
                    idx = random.randint(0, 15)
                    volt_flat[idx] = random.randint(0, 39)
                    violations = _count_violations(volt_flat)
    
    # Strategy 3: If still no solution, try a different base structure
    # Use a different voltage assignment pattern
    if candidate_volt_flat is None:
        # Try volt[i][j] = (i * j * c + i * a + j * b) mod 40 with c even
        for c in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38]:
            for a in range(0, 40, 2):
                for b in range(0, 40, 2):
                    volt_flat = [(i * j * c + i * a + j * b) % MOD for i in range(4) for j in range(4)]
                    if _check_all_walks(volt_flat):
                        candidate_volt_flat = volt_flat
                        break
                if candidate_volt_flat:
                    break
            if candidate_volt_flat:
                break
    
    if candidate_volt_flat is not None:
        # Convert flat to 2D for building
        voltages = [[candidate_volt_flat[i * 4 + j] for j in range(4)] for i in range(4)]
        n, edges = _build_lift(voltages)
        # Final verification with exact girth check
        if _verify_girth_exact(n, edges, g, deadline):
            return n, edges
    
    # Fallback: use LUW construction
    return _construct_luw(4, 10, deadline)


def _construct_luw(k, g, deadline):
    """LUW construction as fallback."""
    q = k
    d_candidates = []
    for d in range(3, 20):
        if d % 2 == 1:
            if d + 5 >= g:
                d_candidates.append(d)
        else:
            if d + 4 >= g:
                d_candidates.append(d)
    d = min(d_candidates) if d_candidates else max(3, g - 5)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rules_cycle = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = rules_cycle[r]
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield (1 - side), tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges
