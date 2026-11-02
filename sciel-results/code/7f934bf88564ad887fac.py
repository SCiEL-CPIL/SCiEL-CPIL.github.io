"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    if k == 4 and g == 10:
        # Try to find a 320-vertex cage via voltage lift of K_{4,4} over Z_40
        voltages = _search_voltage_matrix(deadline - (time.time() - start_time) - 10.0)
        if voltages is not None:
            return _build_lift(voltages)
        
        # Fallback: LUW graph with d=4 (512 vertices)
        return _construct_luw(4, 10, 4)
    
    # General fallback: LUW construction
    return _construct_luw(k, g, None)

def _construct_luw(k, g, forced_d):
    q = k
    if forced_d is not None:
        d = forced_d
    else:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def _build_lift(voltages):
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

def _count_violations(voltages):
    count = 0
    # 4-cycles
    for i in range(4):
        for k in range(4):
            if i == k: continue
            for j in range(4):
                for l in range(4):
                    if j == l: continue
                    if (voltages[i][j] - voltages[k][j] + voltages[k][l] - voltages[i][l]) % 40 == 0:
                        count += 1
    # 6-cycles
    for i in range(4):
        for j in range(4):
            for k in range(4):
                if k == i: continue
                for l in range(4):
                    if l == j: continue
                    for m in range(4):
                        if m == k or m == i: continue
                        for n in range(4):
                            if n == l or n == j: continue
                            s = (voltages[i][j] - voltages[k][j] + voltages[k][l] 
                                 - voltages[m][l] + voltages[m][n] - voltages[i][n]) % 40
                            if s == 0:
                                count += 1
    # 8-cycles (simplified check: all 4 A and 4 B vertices)
    # We check a subset of 8-cycles to speed up search
    for perm_a_rest in itertools.permutations([2, 3]):
        for perm_b in itertools.permutations(range(4)):
            perm_a = [0, 1, perm_a_rest[0], perm_a_rest[1]]
            s = 0
            for idx in range(4):
                a_cur = perm_a[idx]
                b_cur = perm_b[idx]
                a_next = perm_a[(idx + 1) % 4]
                s += voltages[a_cur][b_cur]
                s -= voltages[a_next][b_cur]
            if s % 40 == 0:
                count += 1
    return count

def _search_voltage_matrix(time_limit):
    deadline = time.time() + time_limit
    best_count = float('inf')
    best_volt = None
    
    for _ in range(50): # 50 restarts
        if time.time() > deadline: break
        volt = [[random.randint(0, 39) for _ in range(4)] for _ in range(4)]
        current_count = _count_violations(volt)
        if current_count == 0:
            return volt
        best_count = min(best_count, current_count)
        best_volt = [row[:] for row in volt]
        
        # Local search
        for _ in range(1000):
            if time.time() > deadline: break
            i = random.randint(0, 3)
            j = random.randint(0, 3)
            old_val = volt[i][j]
            new_val = random.randint(0, 39)
            if new_val == old_val: continue
            volt[i][j] = new_val
            new_count = _count_violations(volt)
            if new_count <= current_count:
                current_count = new_count
                if current_count == 0:
                    return volt
                best_count = min(best_count, current_count)
                best_volt = [row[:] for row in volt]
            else:
                volt[i][j] = old_val
                
    if best_count == 0:
        return best_volt
    return None
# EVOLVE-BLOCK-END
