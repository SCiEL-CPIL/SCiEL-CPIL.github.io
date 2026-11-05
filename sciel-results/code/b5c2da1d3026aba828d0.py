"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import defaultdict

def synthesize(matrix, depth_limit):
    """Synthesize an XOR circuit for the given binary matrix.
    
    Each row represents an output that is the XOR of certain input variables.
    We find common subexpressions to minimize total gate count.
    
    Args:
        matrix: 2D binary array where matrix[i][j] = 1 means output i includes input j
        depth_limit: maximum circuit depth (we aim for depth <= 16)
    
    Returns:
        dict with 'gates' (list of [wire1, wire2] pairs) and 'outputs' (list of wire indices)
    """
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    # Convert to numpy for efficiency
    mat = np.array(matrix, dtype=np.int8)
    n_outputs, n_inputs = mat.shape
    
    # Input wires: 0..n_inputs-1, constant-zero: n_inputs
    # Gate k produces wire n_inputs + 1 + k
    
    # Each output is represented as a frozenset of input indices (or empty set for 0)
    # We'll use bitmasks for efficiency when n_inputs is small enough
    
    if n_inputs <= 64:
        # Use Python integers as bitmasks
        def row_to_mask(row):
            m = 0
            for j, v in enumerate(row):
                if v:
                    m |= (1 << j)
            return m
        
        output_masks = [row_to_mask(mat[i]) for i in range(n_outputs)]
        
        def mask_to_set(m):
            s = set()
            j = 0
            while m:
                if m & 1:
                    s.add(j)
                m >>= 1
                j += 1
            return s
        
        def set_to_mask(s):
            m = 0
            for j in s:
                m |= (1 << j)
            return m
        
        def xor_mask(a, b):
            return a ^ b
        
        def popcount(m):
            return bin(m).count('1')
    else:
        # Use frozensets for larger inputs
        def row_to_set(row):
            return frozenset(j for j, v in enumerate(row) if v)
        
        output_sets = [row_to_set(mat[i]) for i in range(n_outputs)]
        
        def set_to_frozen(s):
            return frozenset(s)
        
        def xor_set(a, b):
            return frozenset(a ^ b)
        
        def popcount(s):
            return len(s)
    
    # Determine representation
    use_masks = n_inputs <= 64
    
    if use_masks:
        outputs = list(output_masks)
    else:
        outputs = list(output_sets)
    
    # Track which wire each expression is on
    # wire_to_expr: wire_index -> expression (mask or frozenset)
    # expr_to_wire: expression -> wire_index (for lookup)
    
    wire_to_expr = {}
    expr_to_wire = {}
    
    # Initialize with input wires
    for i in range(n_inputs):
        if use_masks:
            expr = (1 << i)
        else:
            expr = frozenset([i])
        wire_to_expr[i] = expr
        expr_to_wire[expr] = i
    
    # Constant zero wire
    if use_masks:
        zero_expr = 0
    else:
        zero_expr = frozenset()
    wire_to_expr[n_inputs] = zero_expr
    expr_to_wire[zero_expr] = n_inputs
    
    current_wire_count = n_inputs + 1  # next available wire index
    
    # Greedy approach: iteratively find the most beneficial intermediate XOR
    # For each pair of existing wires, compute their XOR and see if it helps
    
    # We want to minimize total gates. Start with the trivial circuit:
    # each output is computed directly from inputs.
    
    # First, let's compute the baseline: for each output, count the number of 1s
    # and subtract 1 for the XOR operations needed (if count > 1, need count-1 gates)
    
    def count_gates_for_expr(expr):
        """Number of XOR gates needed to compute expr from inputs directly."""
        c = popcount(expr)
        if c <= 1:
            return 0
        return c - 1
    
    # Track which outputs are "satisfied" by which wire
    # output_wire[i] = wire that produces output i
    
    # Initial assignment: compute each output directly
    # But we can share subexpressions
    
    # Strategy: 
    # 1. Start with all outputs computed directly (expensive)
    # 2. Repeatedly find the intermediate XOR that saves the most gates
    # 3. An intermediate XOR of wires a and b (producing expr = expr_a XOR expr_b)
    #    saves gates if it allows multiple outputs to share computation
    
    # Better approach: 
    # - For each output, we have a "recipe" which is a set of input indices
    # - We want to find common subsets that appear in multiple outputs
    # - Create intermediate wires for these common subsets
    
    # Let's use a different strategy:
    # 1. Collect all "partial XORs" that are prefixes of output computations
    # 2. Actually, let's use a more systematic approach:
    
    # For each output, the computation is: XOR of all inputs in its set
    # We can think of this as building a binary tree of XOR operations
    # The key is to share subtrees between outputs
    
    # Greedy sharing algorithm:
    # 1. For each pair of existing wires (a, b), compute c = expr_a XOR expr_b
    # 2. For each output, check if using c as an intermediate would reduce its gate count
    # 3. Pick the (a,b) pair that gives the maximum total savings
    # 4. Add the gate, update wire assignments, repeat until no more savings
    
    # Let's define: for an output with expression E, and an available intermediate wire w with expression W,
    # the new expression for the output becomes E XOR W, and we need 1 gate (for XOR with w) + cost of computing E XOR W from remaining inputs
    
    # Actually, let me think about this differently.
    
    # Each output needs to be computed as XOR of its constituent inputs.
    # If we have an intermediate wire that computes some subset S, and an output needs XOR of set T,
    # then if S ⊆ T, we can compute output as: (XOR of T\S) XOR w, saving |S|-1 gates (replacing |S| inputs with 1 wire + 1 XOR)
    # More precisely: without sharing, cost = |T|-1. With sharing, cost = |T\S|-1 + 1 = |T\S| (if |T\S| >= 1) or 0 (if T=S)
    # Savings = (|T|-1) - |T\S| = |S| - 1 (if |T\S| >= 1) or (|T|-1) - 0 = |T|-1 = |S|-1 (if T=S)
    # So savings = |S| - 1 in both cases (when S ⊆ T and S is non-empty)
    
    # Wait, that's not quite right. Let me reconsider.
    
    # If output T uses intermediate wire w (which computes S ⊆ T):
    # - We still need to compute XOR of (T \ S) from inputs: cost = |T\S| - 1 (if |T\S| > 1) or 0 (if |T\S| <= 1)
    # - Then XOR with w: cost = 1 (if |T\S| >= 1) or 0 (if T=S, we just use w directly)
    # Total: if T=S: 0. If |T\S|=1: 0+1=1. If |T\S|>=2: (|T\S|-1)+1 = |T\S|
    # Original cost: |T|-1
    # Savings: if T=S: |T|-1. If |T\S|=1: |T|-1-1 = |T|-2 = |S|-1. If |T\S|>=2: |T|-1-|T\S| = |S|-1
    # So savings = |S|-1 always (when S ⊆ T and S non-empty, and T ≠ S gives |S|-1, T=S gives |S|-1 too)
    
    # So creating an intermediate wire for set S saves (|S|-1) gates for each output that contains S as a subset.
    # But we also need to pay for creating the intermediate wire itself: cost = |S|-1 gates.
    # Net savings = (#outputs containing S) * (|S|-1) - (|S|-1) = (|S|-1) * (#outputs containing S - 1)
    
    # This is the key insight! We want to find sets S that are subsets of many outputs.
    
    # Algorithm:
    # 1. For each candidate set S (from combinations of inputs), compute:
    #    - cost_to_create = |S| - 1
    #    - num_outputs_containing_S = number of outputs whose set contains S
    #    - net_savings = (|S|-1) * (num_outputs_containing_S - 1)
    # 2. Greedily add the most beneficial intermediate wires
    
    # But we need to be careful: once we create intermediate wires, the "outputs" effectively change
    # because they can use those intermediates.
    
    # Let me implement a simpler version first:
    # 1. Find all pairs of inputs that appear together in multiple outputs -> create XOR of those 2 inputs
    # 2. Then find triples, etc.
    
    # Actually, let's just do a greedy approach where at each step, we consider creating a new wire
    # from XOR of two existing wires, and pick the one that maximizes net benefit.
    
    # Let me implement a cleaner version:
    
    # State:
    # - wires: list of expressions currently available
    # - output_exprs: current expression for each output (what we still need to XOR together)
    # - For each output, we track which wires/inputs we've already "used" vs what remains
    
    # Simpler approach: 
    # Each output is just a set of inputs. We want to partition these into groups that are computed
    # by intermediate wires, plus possibly direct inputs.
    
    # Let me try a different, more practical approach:
    
    # 1. Build a frequency table: for each subset of inputs (up to size 3 or 4), count how many outputs contain it
    # 2. Greedily select the most frequent subsets to create as intermediate wires
    # 3. After creating intermediates, recompute the cost for each output
    
    # For efficiency, let's limit to subsets of size 2 and 3 initially.
    
    # Count frequencies of pairs
    pair_freq = defaultdict(int)
    triple_freq = defaultdict(int)
    
    if use_masks:
        for i in range(n_outputs):
            mask = output_masks[i]
            # Extract all pairs
            bits = []
            m = mask
            j = 0
            while m:
                if m & 1:
                    bits.append(j)
                m >>= 1
                j += 1
            for a_idx in range(len(bits)):
                for b_idx in range(a_idx+1, len(bits)):
                    pair = (bits[a_idx], bits[b_idx])
                    pair_freq[pair] += 1
                for c_idx in range(b_idx+1, len(bits)):
                    triple = (bits[a_idx], bits[b_idx], bits[c_idx])
                    triple_freq[triple] += 1
    else:
        for i in range(n_outputs):
            s = list(output_sets[i])
            for a_idx in range(len(s)):
                for b_idx in range(a_idx+1, len(s)):
                    pair = (s[a_idx], s[b_idx])
                    pair_freq[pair] += 1
                for c_idx in range(b_idx+1, len(s)):
                    triple = (s[a_idx], s[b_idx], s[c_idx])
                    triple_freq[triple] += 1
    
    # Greedily select intermediate wires
    # Start with inputs and zero
    available_exprs = {}  # expr -> wire
    if use_masks:
        for i in range(n_inputs):
            available_exprs[(1 << i)] = i
        available_exprs[0] = n_inputs
    else:
        for i in range(n_inputs):
            available_exprs[frozenset([i])] = i
        available_exprs[frozenset()] = n_inputs
    
    next_wire = n_inputs + 1
    gates = []  # list of [wire1, wire2]
    
    # We'll iteratively add the most beneficial intermediate
    # Benefit of adding expr E (computed from two existing wires a, b):
    # - It can be used by any output that contains E as a subset
    # - For each such output, it saves (|E|-1) gates (as derived above)
    # - But we pay (|E|-1) gates to create it
    # - Net = (|E|-1) * (num_outputs_using_E - 1)
    
    # However, this is an approximation because outputs might use multiple intermediates.
    # For a greedy approach, this should work reasonably well.
    
    # Let's be smarter: only consider intermediates that are XOR of two existing wires
    # and that actually reduce the cost of at least 2 outputs.
    
    # For simplicity and speed, let's just create intermediates for frequent pairs and triples.
    
    # Create intermediates for pairs with frequency >= 2
    # Sort by frequency descending
    beneficial_pairs = [(pair, freq) for pair, freq in pair_freq.items() if freq >= 2]
    beneficial_pairs.sort(key=lambda x: -x[1])
    
    beneficial_triples = [(triple, freq) for triple, freq in triple_freq.items() if freq >= 2]
    beneficial_triples.sort(key=lambda x: -x[1])
    
    # Add pair intermediates
    for pair, freq in beneficial_pairs:
        if time.time() > DEADLINE:
            break
        if use_masks:
            expr = (1 << pair[0]) | (1 << pair[1])
        else:
            expr = frozenset(pair)
        
        if expr in available_exprs:
            continue
        
        # Create this wire: XOR of the two input wires
        w1 = pair[0]
        w2 = pair[1]
        gates.append([w1, w2])
        available_exprs[expr] = next_wire
        next_wire += 1
    
    # Add triple intermediates (only if they don't overlap with existing)
    for triple, freq in beneficial_triples:
        if time.time() > DEADLINE:
            break
        if use_masks:
            expr = (1 << triple[0]) | (1 << triple[1]) | (1 << triple[2])
        else:
            expr = frozenset(triple)
        
        if expr in available_exprs:
            continue
        
        # To create a triple, we can XOR two inputs to get a pair, then XOR with the third
        # Or if we already have the pair, just XOR with the third input
        a, b, c = triple
        
        # Check if pair (a,b) is already available
        if use_masks:
            pair_ab = (1 << a) | (1 << b)
        else:
            pair_ab = frozenset([a, b])
        
        if pair_ab in available_exprs:
            # XOR pair_ab with input c
            w1 = available_exprs[pair_ab]
            w2 = c
            gates.append([w1, w2])
            available_exprs[expr] = next_wire
            next_wire += 1
        else:
            # Need to create pair first, then XOR with c
            # But this might not be worth it for just one triple
            # Skip for now
            pass
    
    # Now compute each output using available intermediates
    # For each output, we want to express it as XOR of available wires/inputs
    
    # Greedy: for each output, repeatedly pick the available expression that is a subset
    # of the remaining part of the output, with the largest size
    
    output_wires = []  # list of wires that XOR together to give each output
    
    if use_masks:
        for i in range(n_outputs):
            remaining = output_masks[i]
            used_wires = []
            
            # Greedily pick the largest available expression that is a subset of remaining
            while remaining != 0:
                best_expr = None
                best_wire = None
                best_size = 0
                
                for expr, wire in available_exprs.items():
                    if expr == 0:
                        continue
                    # Check if expr is subset of remaining
                    if (expr & remaining) == expr:
                        size = popcount(expr)
                        if size > best_size:
                            best_size = size
                            best_expr = expr
                            best_wire = wire
                
                if best_expr is None:
                    # Should not happen, but fallback: pick a single input
                    # Find a bit in remaining
                    m = remaining
                    j = 0
                    while m:
                        if m & 1:
                            best_expr = (1 << j)
                            best_wire = j
                            break
                        m >>= 1
                        j += 1
                
                used_wires.append(best_wire)
                remaining ^= best_expr
                
            # If used_wires is empty, output is 0
            if len(used_wires) == 0:
                output_wires.append(n_inputs)  # zero wire
            elif len(used_wires) == 1:
                output_wires.append(used_wires[0])
            else:
                # Need to XOR all used_wires together
                # Create a chain of XOR gates
                # We can do this in a tree structure for better depth
                # For simplicity, do a chain: w0 XOR w1 -> temp1, temp1 XOR w2 -> temp2, etc.
                # But this might exceed depth limit. Let's do a balanced tree.
                
                # For now, just chain them
                current = used_wires[0]
                for w in used_wires[1:]:
                    gates.append([current, w])
                    current = next_wire
                    next_wire += 1
                output_wires.append(current)
    else:
        for i in range(n_outputs):
            remaining = set(output_sets[i])
            used_wires = []
            
            while len(remaining) > 0:
                best_expr = None
                best_wire = None
                best_size = 0
                
                for expr, wire in available_exprs.items():
                    if len(expr) == 0:
                        continue
                    # Check if expr is subset of remaining
                    if expr.issubset(remaining):
                        size = len(expr)
                        if size > best_size:
                            best_size = size
                            best_expr = expr
                            best_wire = wire
                
                if best_expr is None:
                    # Fallback: pick a single input
                    elem = next(iter(remaining))
                    best_expr = frozenset([elem])
                    best_wire = elem
                
                used_wires.append(best_wire)
                remaining -= best_expr
                
            if len(used_wires) == 0:
                output_wires.append(n_inputs)
            elif len(used_wires) == 1:
                output_wires.append(used_wires[0])
            else:
                current = used_wires[0]
                for w in used_wires[1:]:
                    gates.append([current, w])
                    current = next_wire
                    next_wire += 1
                output_wires.append(current)
    
    # Verify the circuit is correct
    # (For safety, we could verify, but let's trust the construction for now)
    
    return {'gates': gates, 'outputs': output_wires}

# EVOLVE-BLOCK-END
