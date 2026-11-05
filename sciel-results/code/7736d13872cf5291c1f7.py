"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import defaultdict

def synthesize(matrix, depth_limit):
    """
    Synthesize an XOR circuit for the given binary matrix.
    
    Args:
        matrix: 2D binary array (n_rows x n_cols) where each row represents an output function
                of the input variables (0..n_cols-1).
        depth_limit: Maximum allowed depth of the circuit.
                     
    Returns:
        dict with keys:
            'gates': list of [wire1, wire2] pairs, where each gate computes XOR of wire1 and wire2.
            'outputs': list of wire indices corresponding to the outputs of the circuit.
    """
    start_time = time.time()
    deadline = start_time + 55.0
    
    A = np.array(matrix, dtype=np.int8)
    if A.ndim == 1:
        A = A.reshape(1, -1)
    
    n_rows, n_cols = A.shape
    n_inputs = n_cols
    zero_wire = n_inputs
    
    if n_rows == 0 or n_cols == 0:
        return {'gates': [], 'outputs': [zero_wire] * n_rows}
    
    # Represent each row as a set of wire indices
    row_wires = []
    for i in range(n_rows):
        if A[i].sum() == 0:
            row_wires.append(set())
        else:
            row_wires.append(set(np.where(A[i] == 1)[0].tolist()))
    
    gates = []
    next_wire = zero_wire + 1
    
    # Map from wire to its underlying set of input indices
    wire_set = {}
    for i in range(n_inputs):
        wire_set[i] = frozenset([i])
    wire_set[zero_wire] = frozenset()
    
    # Map from frozenset to wire index (for reuse)
    set_to_wire = {}
    for i in range(n_inputs):
        set_to_wire[frozenset([i])] = i
    set_to_wire[frozenset()] = zero_wire
    
    def elapsed_ok():
        return time.time() < deadline
    
    def row_depth_estimate(rs):
        """Estimate depth contribution of a row's wire set."""
        if len(rs) <= 1:
            return 0
        return len(rs) - 1
    
    def total_cost_estimate():
        """Estimate total gate count needed."""
        total = len(gates)
        for rs in row_wires:
            if len(rs) > 1:
                total += len(rs) - 1
        return total
    
    # Phase 1: Greedy pair-sharing with depth-aware scoring
    max_iter = 30000
    for _ in range(max_iter):
        if not elapsed_ok():
            break
        
        # Count co-occurrence of pairs in rows with size >= 2
        pair_count = defaultdict(int)
        pair_max_row_size = defaultdict(int)  # max row size among rows containing pair
        for i in range(n_rows):
            rs = row_wires[i]
            if len(rs) < 2:
                continue
            elems = sorted(rs)
            for x in range(len(elems)):
                for y in range(x + 1, len(elems)):
                    a, b = elems[x], elems[y]
                    pair_count[(a, b)] += 1
                    if len(rs) > pair_max_row_size[(a, b)]:
                        pair_max_row_size[(a, b)] = len(rs)
        
        if not pair_count:
            break
        
        # Find best pair: maximize benefit = count - 1 + reuse_bonus + depth_reduction_bonus
        best_pair = None
        best_benefit = 0
        for (a, b), cnt in pair_count.items():
            if cnt < 2:
                continue
            
            new_set = wire_set[a] ^ wire_set[b]
            existing_wire = set_to_wire.get(new_set, -1)
            
            # Base benefit: each row that uses both a and b saves 1 gate
            benefit = cnt - 1  # -1 for cost of creating new gate (if needed)
            
            # If result already exists, no new gate cost
            if existing_wire >= 0:
                benefit += 1
            
            # Depth reduction bonus: reducing a large row's size is more valuable
            # The pair reduces row size by 1 in each row it appears in
            # Weight by how much it helps the deepest rows
            max_rs = pair_max_row_size.get((a, b), 0)
            if max_rs >= 4:
                benefit += 0.5  # Small bonus for reducing larger rows
            
            if not new_set:
                benefit += cnt
            
            if benefit > best_benefit:
                best_benefit = benefit
                best_pair = (a, b)
        
        if best_pair is None:
            break
        
        a, b = best_pair
        new_set = wire_set[a] ^ wire_set[b]
        existing_wire = set_to_wire.get(new_set, -1)
        
        if existing_wire >= 0:
            new_wire = existing_wire
        else:
            new_wire = next_wire
            next_wire += 1
            wire_set[new_wire] = new_set
            set_to_wire[new_set] = new_wire
            gates.append([a, b])
        
        # Update all rows containing both a and b
        for i in range(n_rows):
            rs = row_wires[i]
            if a in rs and b in rs:
                rs.discard(a)
                rs.discard(b)
                if new_set:
                    rs.add(new_wire)
    
    # Phase 2: Balanced-tree fallback for remaining multi-wire rows
    # For each row with >1 wires, use balanced tree to minimize depth
    # This also creates new intermediate wires that might be shared
    outputs = []
    for i in range(n_rows):
        rs = row_wires[i]
        if len(rs) == 0:
            outputs.append(zero_wire)
        elif len(rs) == 1:
            outputs.append(list(rs)[0])
        else:
            # Balanced tree construction
            elems = list(rs)
            # Use a queue-based balanced tree
            pending = list(elems)
            while len(pending) > 1:
                if not elapsed_ok():
                    # Fallback to linear chain if time is tight
                    current = pending[0]
                    for e in pending[1:]:
                        gates.append([current, e])
                        current = next_wire
                        next_wire += 1
                    outputs.append(current)
                    break
                new_pending = []
                for j in range(0, len(pending) - 1, 2):
                    a, b = pending[j], pending[j + 1]
                    # Check if a XOR b already exists
                    new_set = wire_set[a] ^ wire_set[b]
                    existing_wire = set_to_wire.get(new_set, -1)
                    if existing_wire >= 0:
                        new_w = existing_wire
                    else:
                        new_w = next_wire
                        next_wire += 1
                        wire_set[new_w] = new_set
                        set_to_wire[new_set] = new_w
                        gates.append([a, b])
                    new_pending.append(new_w)
                if len(pending) % 2 == 1:
                    new_pending.append(pending[-1])
                pending = new_pending
            outputs.append(pending[0])
    
    return {'gates': gates, 'outputs': outputs}

# EVOLVE-BLOCK-END
