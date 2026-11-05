import time
import numpy as np
from collections import defaultdict

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 85
    
    matrix = np.array(matrix, dtype=int)
    m, n = matrix.shape
    
    baseline = sum(max(0, int(np.sum(row)) - 1) for row in matrix)
    if baseline == 0:
        return {'gates': [], 'outputs': [n] * m}
    
    output_sets = []
    for i in range(m):
        row = matrix[i]
        indices = np.where(row == 1)[0]
        output_sets.append(frozenset(indices.tolist()))
    
    wire_expr = {i: frozenset([i]) for i in range(n)}
    wire_expr[n] = frozenset()
    
    output_exprs = [frozenset(s) for s in output_sets]
    
    gates = []
    next_wire = n + 1
    
    while True:
        if time.time() > deadline:
            break
        
        pair_count = defaultdict(int)
        for oe in output_exprs:
            if len(oe) >= 2:
                wires = list(oe)
                for i in range(len(wires)):
                    for j in range(i+1, len(wires)):
                        u, v = wires[i], wires[j]
                        if u > v:
                            u, v = v, u
                        pair_count[(u, v)] += 1
        
        best_gain = 0
        best_pair = None
        for (u, v), cnt in pair_count.items():
            net_gain = cnt - 1
            if net_gain > best_gain:
                best_gain = net_gain
                best_pair = (u, v)
        
        if best_pair is None:
            break
        
        u, v = best_pair
        new_expr = wire_expr[u].symmetric_difference(wire_expr[v])
        gates.append((u, v))
        wire_expr[next_wire] = new_expr
        
        new_output_exprs = []
        for oe in output_exprs:
            if u in oe and v in oe:
                new_oe = set(oe)
                new_oe.remove(u)
                new_oe.remove(v)
                new_oe.add(next_wire)
                new_output_exprs.append(frozenset(new_oe))
            else:
                new_output_exprs.append(oe)
        output_exprs = new_output_exprs
        next_wire += 1
    
    final_gates = list(gates)
    final_outputs = []
    current_wire = next_wire
    
    for oe in output_exprs:
        if len(oe) == 0:
            final_outputs.append(n)
        elif len(oe) == 1:
            final_outputs.append(list(oe)[0])
        else:
            wires = sorted(oe)
            result = wires[0]
            for w in wires[1:]:
                final_gates.append((result, w))
                result = current_wire
                current_wire += 1
            final_outputs.append(result)
    
    return {'gates': final_gates, 'outputs': final_outputs}
