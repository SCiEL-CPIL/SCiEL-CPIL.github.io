import time
import random
import numpy as np
from collections import defaultdict

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    if m == 0:
        return {'gates': [], 'outputs': []}

    # Convert rows to bitsets for efficient GF(2) manipulation
    # Wire i corresponds to bit i. 
    # Input wires: 0..n-1
    # Constant zero: n
    # Gate k output: n+1+k
    row_bits = []
    for i in range(m):
        bits = 0
        for j in range(n):
            if matrix[i][j]:
                bits |= (1 << j)
        row_bits.append(bits)

    # If all zeros
    if not any(row_bits):
        return {'gates': [], 'outputs': [n] * m}

    # Helper to verify circuit
    def _verify(circuit):
        gates = circuit['gates']
        outputs = circuit['outputs']
        num_wires = n + 1 + len(gates)
        wire_val = [0] * num_wires
        for i in range(n):
            wire_val[i] = 1 << i
        wire_val[n] = 0

        for k, (a, b) in enumerate(gates):
            if a >= num_wires or b >= num_wires:
                return False
            wire_val[n + 1 + k] = wire_val[a] ^ wire_val[b]

        for i in range(m):
            expected = row_bits[i]
            if outputs[i] >= num_wires:
                return False
            if wire_val[outputs[i]] != expected:
                return False
        return True

    def _compute_depth(gates, outputs):
        if not gates:
            return 0
        num_wires = n + 1 + len(gates)
        wire_depth = [0] * num_wires
        for k, (a, b) in enumerate(gates):
            wire_depth[n + 1 + k] = max(wire_depth[a], wire_depth[b]) + 1
        max_depth = 0
        for w in outputs:
            if w < num_wires:
                max_depth = max(max_depth, wire_depth[w])
        return max_depth

    def _baseline_circuit():
        """Independent XOR of inputs for each output."""
        gates = []
        outputs = []
        next_wire = n + 1
        for i in range(m):
            bits = row_bits[i]
            if bits == 0:
                outputs.append(n)
            else:
                indices = [j for j in range(n) if (bits >> j) & 1]
                if len(indices) == 1:
                    outputs.append(indices[0])
                else:
                    prev = indices[0]
                    for idx in indices[1:]:
                        gates.append([prev, idx])
                        prev = next_wire
                        next_wire += 1
                    outputs.append(prev)
        return {'gates': gates, 'outputs': outputs}

    def _randomized_gaussian_elimination(seed, strategy='random'):
        """
        Performs Gaussian elimination over GF(2) using bitsets.
        Returns a list of expressions (bitsets) representing the rows in terms of a new basis,
        and a list of gates used to create that basis.
        
        Strategy:
        - 'random': Pick a random pivot row.
        - 'min_nnz': Pick the row with minimum non-zero bits (sparsest) as pivot.
        """
        rng = random.Random(seed)
        exprs = list(row_bits)
        gates = []
        next_wire = n + 1
        
        # We will eliminate columns 0 to n-1
        # We maintain a list of active rows
        active = list(range(m))
        
        # To keep track of what the current expressions mean in terms of original inputs and new gates
        # Actually, standard GE modifies the rows. 
        # If we do x_j = x_j XOR x_i, we are replacing a column.
        # It is easier to work with the dual: express rows in terms of a basis.
        
        # Let's use a different approach: 
        # We want to find a set of intermediate wires g_1, ..., g_k such that each row is a sparse XOR of inputs and g's.
        # This is equivalent to finding a sparse matrix B such that M = B * G, where G is the gate matrix.
        
        # A simpler heuristic that works well:
        # 1. Identify frequent sub-expressions.
        # 2. But first, let's try to reduce the density of the matrix using random row operations? No, row ops change the outputs.
        
        # Let's stick to the CSE approach but with a better initialization or a more global view.
        
        # Alternative: 
        # Start with the rows.
        # Iteratively, find the pair of wires (a,b) that appears in the most rows.
        # Create g = a ^ b.
        # Replace a,b with g in all rows where both appear.
        # This is what the current program does.
        
        # How to improve?
        # 1. Use a priority queue for pairs.
        # 2. Consider "chains" of XORs.
        # 3. Use a global cost function: sum of popcounts of all expressions.
        
        return None # Placeholder, we will use the CSE directly but optimized

    def _greedy_cse_optimized(time_budget, seed, use_min_nnz_pivot=False):
        """
        Optimized Greedy CSE.
        """
        rng = random.Random(seed)
        exprs = list(row_bits)
        gates = []
        next_wire = n + 1
        
        # Frequency of each wire in the expressions
        # We need to update this efficiently.
        # For m up to 16, n up to 16, bitsets are fast.
        
        max_iter = 500
        for _ in range(max_iter):
            if time.time() > time_budget:
                break
            
            # Count frequency of each wire
            freq = defaultdict(int)
            for e in exprs:
                if e == 0: continue
                # Iterate through set bits
                temp = e
                while temp:
                    lsb = temp & -temp
                    w = lsb.bit_length() - 1
                    freq[w] += 1
                    temp ^= lsb
            
            if not freq:
                break
            
            # Only consider wires with freq >= 2
            candidates = [w for w, c in freq.items() if c >= 2]
            if len(candidates) < 2:
                break
            
            # Find best pair
            best_pair = None
            best_benefit = 1
            best_cost_reduction = 0
            
            # Sort candidates by frequency to prioritize common wires
            candidates.sort(key=lambda w: freq[w], reverse=True)
            top_k = min(len(candidates), 40)
            top_candidates = candidates[:top_k]
            
            # To avoid O(K^2 * M) being too slow if K is large, we limit K.
            # With M=16, N=16, K is small.
            
            for i in range(len(top_candidates)):
                a = top_candidates[i]
                a_bit = 1 << a
                for j in range(i + 1, len(top_candidates)):
                    b = top_candidates[j]
                    b_bit = 1 << b
                    
                    benefit = 0
                    for e in exprs:
                        if (e & a_bit) and (e & b_bit):
                            benefit += 1
                    
                    if benefit > best_benefit or (benefit == best_benefit and rng.random() < 0.1):
                        best_benefit = benefit
                        best_pair = (a, b)
            
            if best_pair is None:
                break
            
            a, b = best_pair
            g = next_wire
            next_wire += 1
            gates.append([a, b])
            
            a_bit = 1 << a
            b_bit = 1 << b
            g_bit = 1 << g
            
            for i in range(m):
                e = exprs[i]
                if (e & a_bit) and (e & b_bit):
                    e = (e & ~a_bit & ~b_bit) | g_bit
                    exprs[i] = e

        # Build outputs
        outputs = []
        for i in range(m):
            e = exprs[i]
            if e == 0:
                outputs.append(n)
            else:
                wires = []
                temp = e
                while temp:
                    lsb = temp & -temp
                    w = lsb.bit_length() - 1
                    wires.append(w)
                    temp ^= lsb

                if len(wires) == 1:
                    outputs.append(wires[0])
                elif len(wires) == 0:
                    outputs.append(n)
                else:
                    # XOR all wires together
                    # To minimize depth, we can build a tree instead of a chain
                    # But for gate count, it's the same.
                    # Chain is simpler.
                    prev = wires[0]
                    for w in wires[1:]:
                        gates.append([prev, w])
                        prev = next_wire
                        next_wire += 1
                    outputs.append(prev)

        return {'gates': gates, 'outputs': outputs}

    def _local_polish(circuit, time_budget, seed):
        """
        Try to reduce gate count by removing redundant gates or merging.
        This is a heuristic pass.
        """
        rng = random.Random(seed)
        gates = circuit['gates']
        outputs = circuit['outputs']
        
        # Check if any gate is unused
        used_wires = set(outputs)
        for g in gates:
            used_wires.add(g[0])
            used_wires.add(g[1])
            
        # This is complex to implement correctly without re-verification.
        # Skip for now, rely on CSE quality.
        return circuit

    # First, get a baseline
    baseline = _baseline_circuit()
    best_result = baseline
    best_gate_count = len(baseline['gates'])

    # Try multiple strategies with different seeds
    # The "global vector-space" idea might be implemented by running CSE on different projections or with different tie-breaking.
    
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
    
    # Allocate time for each seed
    # We have ~85s. Let's do 30 seeds. ~2.8s per seed.
    
    for seed in seeds:
        if time.time() > deadline - 2.0:
            break

        try:
            # Use a slightly tighter budget for each run to allow for multiple runs
            run_budget = min(deadline - 2.0, time.time() + 2.5)
            result = _greedy_cse_optimized(run_budget, seed)
            
            if _verify(result):
                d = _compute_depth(result['gates'], result['outputs'])
                if d <= depth_limit:
                    gc = len(result['gates'])
                    if gc < best_gate_count:
                        best_gate_count = gc
                        best_result = result
        except Exception:
            pass

    return best_result
