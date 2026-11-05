"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def _to_int(row):
    """Convert a numpy row (or list) to an integer representation."""
    if isinstance(row, np.ndarray):
        # Treat as bytes: each element is 0 or 1
        val = 0
        for i in range(len(row)):
            if row[i]:
                val |= (1 << i)
        return val
    return int(row)

def _popcount(x):
    return bin(x).count('1')

def _xor_list(lst):
    """XOR all elements in a list of ints."""
    val = 0
    for x in lst:
        val ^= x
    return val

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin before 60s kill
    
    # Parse matrix
    # matrix is expected to be a 2D array of 0s and 1s
    if isinstance(matrix, list):
        matrix = np.array(matrix)
    
    m, n = matrix.shape
    n_cols = n
    
    # Target rows: each is a bitmask of length n_cols
    targets = []
    for i in range(m):
        targets.append(_to_int(matrix[i]))
    
    # Input wires are 0..n-1, constant-zero wire is n
    # Gate k produces wire n+1+k
    # We need to synthesize each target row as a function of inputs
    
    # Baseline: each target row i is synthesized by XOR-ing all input wires j where matrix[i][j]==1
    # This uses popcount(target_i) - 1 gates if popcount > 1, or 0 if popcount == 0, or 0 if popcount == 1 (just wire j)
    # Actually, for a single wire, no gate needed. For k wires, k-1 gates.
    
    # Strategy: Global optimization using iterative greedy sharing
    # We maintain a set of "available" wires (initially inputs 0..n-1 and zero wire n)
    # We want to produce each target. We can create intermediate gates to share common sub-expressions.
    
    # Let's use a different approach:
    # 1. Identify all distinct target rows.
    # 2. For each target, we need to compute it.
    # 3. We can pre-compute intermediate XORs that are useful for multiple targets.
    
    # More sophisticated: 
    # Let's represent the problem as finding a set of gates (XORs of existing wires) that allows us to compute all targets.
    # We start with wires = {0, ..., n-1, n} (n is zero)
    # Each gate adds a new wire which is the XOR of two existing wires.
    
    # Greedy approach:
    # At each step, find a pair of existing wires (a, b) such that their XOR (a^b) is "useful".
    # Usefulness: how many targets can be reduced in complexity if we have this new wire?
    
    # Actually, let's think about it differently.
    # The cost of synthesizing a target t given a set of available wires W is the minimum number of XOR gates needed.
    # If t is in W, cost is 0.
    # If t is the XOR of two wires in W, cost is 1.
    # In general, it's the distance in the XOR graph.
    
    # A simpler heuristic:
    # 1. Start with all input wires.
    # 2. For each target, if it's not directly available, we need to build it.
    # 3. We can share sub-expressions.
    
    # Let's try a different strategy: 
    # Compute the linear basis of the targets.
    # The number of gates needed is at least the rank of the target matrix minus the number of inputs that are directly used?
    # No, that's not quite right.
    
    # Let's implement a practical heuristic:
    # 1. Identify all unique non-zero targets.
    # 2. For each unique target, find the most efficient way to synthesize it.
    # 3. Share intermediate results where possible.
    
    # Actually, the best approach for small n (up to 16) is:
    # 1. Enumerate all possible XOR combinations of small subsets of inputs.
    # 2. Use these as "building blocks".
    # 3. Greedily select building blocks that reduce the total cost.
    
    # Let's start with a simpler but effective method:
    # 1. For each target, compute the set of input indices it depends on.
    # 2. If two targets share a large subset, compute the XOR of that subset once.
    
    # Even simpler: 
    # 1. Sort targets by popcount (descending).
    # 2. For each target, try to express it as XOR of two existing wires (inputs or previously computed gates).
    # 3. If not possible, compute it from scratch using the fewest gates.
    
    # Let's implement a version that uses a "wire pool" and greedily adds gates.
    
    # Available wires: dict from wire_id to its value (int bitmask)
    # wire_id 0..n-1: inputs
    # wire_id n: zero
    # wire_id n+1+k: gate k
    
    wires = {}
    for i in range(n_cols):
        wires[i] = (1 << i)
    wires[n_cols] = 0  # zero wire
    
    next_wire_id = n_cols + 1
    gates = []  # list of (wire_a, wire_b) for each gate
    
    # Targets to synthesize
    remaining_targets = list(targets)
    
    # For each target, we'll track its "cost" given current wires
    # Cost = 0 if target is in wires
    # Cost = 1 if target is XOR of two wires in wires
    # Cost = 2 if target is XOR of a wire and a 2-gate expression, etc.
    
    # We'll use a BFS-like approach to find the cost of each target given current wires.
    # But that might be slow. Let's use a simpler heuristic.
    
    # At each step:
    # 1. For each remaining target, check if it can be formed by XOR of two existing wires.
    # 2. If yes, add a gate for it (cost 1).
    # 3. Otherwise, find the pair of existing wires that minimizes the "remaining popcount" of the target after XOR.
    #    I.e., find (a, b) in wires such that popcount(target ^ a ^ b) is minimized.
    #    Then add gate (a, b), and update target to target ^ a ^ b.
    
    # This is a greedy approach that should work well.
    
    # To make it more efficient, we'll limit the number of gates we consider per target.
    
    max_gates_per_target = 16  # depth limit
    
    while remaining_targets:
        if time.time() > deadline:
            # Return best effort: synthesize remaining targets from scratch
            for t in remaining_targets:
                # Synthesize t from inputs directly
                bits = [i for i in range(n_cols) if (t >> i) & 1]
                for j in range(1, len(bits)):
                    if j == 1:
                        gates.append([bits[0], bits[1]])
                    else:
                        gates.append([next_wire_id - 1, bits[j]])
                    next_wire_id += 1
                if len(bits) == 1:
                    remaining_targets[remaining_targets.index(t)] = bits[0]
                elif len(bits) == 0:
                    remaining_targets[remaining_targets.index(t)] = n_cols
                else:
                    remaining_targets[remaining_targets.index(t)] = next_wire_id - 1
            break
        
        # Find the best gate to add
        best_gate = None
        best_gain = 0
        best_target_idx = -1
        
        # Try all pairs of existing wires
        wire_ids = list(wires.keys())
        wire_vals = {wid: wires[wid] for wid in wire_ids}
        
        # For efficiency, only consider a subset of pairs if there are too many
        if len(wire_ids) > 50:
            # Sample pairs
            import random
            pairs = list(combinations(range(len(wire_ids)), 2))
            if len(pairs) > 1000:
                pairs = random.sample(pairs, 1000)
        else:
            pairs = list(combinations(range(len(wire_ids)), 2))
        
        for i, j in pairs:
            a_id = wire_ids[i]
            b_id = wire_ids[j]
            a_val = wire_vals[a_id]
            b_val = wire_vals[b_id]
            new_val = a_val ^ b_val
            
            if new_val == 0:
                continue
            
            # Check if new_val is already in wires
            if new_val in wire_vals.values():
                continue
            
            # Gain: how much does this reduce the total cost of remaining targets?
            gain = 0
            for idx, t in enumerate(remaining_targets):
                if t == new_val:
                    gain += 1  # Can be done in 1 gate
                else:
                    # Check if t can be formed by XOR of new_val and one existing wire
                    for w_id in wire_ids:
                        if t == new_val ^ wire_vals[w_id]:
                            gain += 0.5  # Can be done in 2 gates instead of more
                            break
            # Also consider if new_val helps reduce popcount
            # For simplicity, just use the above gain
            
            if gain > best_gain:
                best_gain = gain
                best_gate = (a_id, b_id, new_val)
                best_target_idx = -1  # Not tied to a specific target
        
        # If no beneficial gate found, synthesize the target with highest popcount directly
        if best_gate is None or best_gain < 0.1:
            # Pick the target with highest popcount
            max_pop = -1
            max_idx = -1
            for idx, t in enumerate(remaining_targets):
                p = _popcount(t)
                if p > max_pop:
                    max_pop = p
                    max_idx = idx
            
            if max_idx == -1:
                break
            
            t = remaining_targets[max_idx]
            bits = [i for i in range(n_cols) if (t >> i) & 1]
            
            if len(bits) == 0:
                remaining_targets[max_idx] = n_cols
                continue
            elif len(bits) == 1:
                remaining_targets[max_idx] = bits[0]
                continue
            else:
                # Synthesize from scratch
                # Use a simple method: XOR all bits
                # To minimize gates, we can use a tree-like structure
                current = bits[0]
                for j in range(1, len(bits)):
                    gates.append([current, bits[j]])
                    current = next_wire_id
                    wires[current] = wires[bits[0]] if j == 1 else (wires[current - 1] ^ (1 << bits[j]))
                    next_wire_id += 1
                remaining_targets[max_idx] = current
                continue
        
        # Add the best gate
        a_id, b_id, new_val = best_gate
        gates.append([a_id, b_id])
        wires[next_wire_id] = new_val
        next_wire_id += 1
        
        # Update remaining targets: if any target equals new_val, mark it as done
        new_remaining = []
        for t in remaining_targets:
            if t == new_val:
                # This target is now available
                pass  # We'll handle it in the next iteration
            else:
                new_remaining.append(t)
        # Actually, we should remove targets that are now directly available
        remaining_targets = [t for t in remaining_targets if t != new_val]
        
        # Also, we can update targets that can now be formed with one more gate
        # But for simplicity, we'll just continue the loop
    
    # Final step: ensure all targets are mapped to wires
    outputs = []
    for t in targets:
        # Find which wire corresponds to t
        found = False
        for wid, wval in wires.items():
            if wval == t:
                outputs.append(wid)
                found = True
                break
        if not found:
            # Should not happen, but fallback
            bits = [i for i in range(n_cols) if (t >> i) & 1]
            if len(bits) == 0:
                outputs.append(n_cols)
            elif len(bits) == 1:
                outputs.append(bits[0])
            else:
                # Synthesize
                current = bits[0]
                for j in range(1, len(bits)):
                    gates.append([current, bits[j]])
                    current = next_wire_id
                    wires[current] = wires.get(current - 1, 0) ^ (1 << bits[j])
                    next_wire_id += 1
                outputs.append(current)
    
    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END
