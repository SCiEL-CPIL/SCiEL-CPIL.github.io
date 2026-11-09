"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from typing import List, Tuple

def _bit_vector_to_int(vec):
    """Convert a numpy bit vector to an integer for hashing."""
    return int(np.dot(vec, 1 << np.arange(len(vec))))

def _int_to_bit_vector(val, length):
    """Convert an integer back to a numpy bit vector."""
    return np.array([(val >> i) & 1 for i in range(length)], dtype=np.int8)

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    # Ensure matrix is numpy array
    if isinstance(matrix, list):
        matrix = np.array(matrix, dtype=np.int8)
    
    n_rows, n_cols = matrix.shape
    n_input = n_cols
    n_output = n_rows
    
    # We need to compute n_output linear functions of n_input variables.
    # Each output is a subset of inputs XORed together.
    # Baseline: each output is computed independently using its non-zero coefficients.
    # This takes sum(weight) gates if we use a tree or just chain (chain is worse for depth, tree is better).
    # Actually, the baseline usually implies just the number of 1s in the matrix if we assume
    # each 1 is a gate, or more accurately, the cost to build each linear combination.
    # If we build each output independently, the cost is sum of weights of each row.
    # But we can share sub-expressions.
    
    # Let's represent each linear function as a set of input indices or a bitmask.
    # Since n_cols is small (max 16), we can use integers to represent the linear functions.
    # bit i set means input i is included.
    
    current_funcs = [0] * n_output
    for i in range(n_output):
        current_funcs[i] = 0
        for j in range(n_input):
            if matrix[i, j]:
                current_funcs[i] |= (1 << j)
                
    # We also have the constant zero wire (index n_input).
    # We will build a circuit.
    # Available wires:
    # 0..n_input-1: inputs
    # n_input: zero
    # n_input+1..: intermediate gates
    
    # We want to minimize total gates.
    # Strategy:
    # 1. Identify all unique linear functions needed (rows of matrix).
    # 2. Try to find a common sub-expression structure.
    
    # A simple heuristic:
    # If a function is simple (weight 1), it's just an input. Cost 0.
    # If weight 0, it's zero. Cost 0.
    # Otherwise, we need gates.
    
    # Let's use a greedy approach to build the circuit.
    # We maintain a set of "available" linear functions (wires) and their costs.
    # Initially, all inputs and zero are available with cost 0.
    # We want to produce the target functions.
    
    # However, the "baseline" mentioned in the prompt likely refers to the independent cost.
    # Independent cost = sum of weights of all rows? Or sum of (weight - 1)?
    # If weight is 1, we use the input directly.
    # If weight is k > 1, we need k-1 gates to combine k inputs (tree structure).
    # So baseline cost = sum(max(0, weight(row) - 1)) for all rows?
    # Let's assume the baseline is the sum of weights minus the number of non-zero rows.
    
    # Let's try to improve upon independent synthesis by sharing.
    # This is equivalent to finding a straight-line program (SLP) for the linear forms.
    
    # Since depth is limited to 16, and n_cols <= 16, the depth of a balanced tree for 16 inputs is 4.
    # So depth is not a tight constraint for the inputs themselves.
    
    # Algorithm:
    # 1. Map each target function to a bitmask.
    # 2. If a function is already an input or zero, mark it as resolved.
    # 3. For the rest, we try to find pairs of available functions that can be XORed to produce another available or target function.
    #    This is a form of common subexpression elimination.
    
    # Let's implement a greedy CSE (Common Subexpression Elimination).
    # State:
    # - available: dict mapping bitmask -> wire_index
    # - targets: dict mapping bitmask -> list of output_indices
    # - gates: list of [wire1, wire2]
    
    available = {}
    for i in range(n_input):
        available[1 << i] = i
    available[0] = n_input  # zero wire
    
    targets = {}
    for i in range(n_output):
        mask = current_funcs[i]
        if mask not in targets:
            targets[mask] = []
        targets[mask].append(i)
    
    # Remove targets that are already available
    for mask in list(targets.keys()):
        if mask in available:
            del targets[mask]
            
    # If no targets left, we are done.
    
    # Greedy loop:
    # While targets exist and time < deadline:
    #   Find a pair (a, b) in available such that a^b is a target or can help reduce targets.
    #   Actually, just finding a^b that is a target is the direct win.
    #   If no direct win, maybe we should create a new intermediate?
    #   Creating an intermediate is only useful if it helps reduce the cost of multiple targets.
    
    # Let's stick to the "direct win" strategy first:
    # If two available functions XOR to a target, create that gate.
    
    # But what if no two available functions XOR to a target?
    # Then we must create new intermediates.
    # Which intermediates?
    # Heuristic: pick the most "frequent" bits or something similar.
    # Or, just fall back to independent synthesis for the remaining targets if no sharing is found quickly.
    
    # Let's try a slightly more sophisticated approach:
    # 1. Check for direct XOR matches.
    # 2. If none, pick the target with the smallest weight > 1.
    #    Decompose it into two parts?
    #    Actually, if we can't find sharing, independent synthesis is optimal for that specific target relative to current availables?
    #    Not necessarily, but it's a safe fallback.
    
    # Let's implement the direct matching loop.
    
    # To speed up matching, we can keep a list of available masks.
    # For each target, check if it can be formed by XOR of two availables.
    
    # Optimization: Precompute a set of available masks for O(1) lookup?
    # We need to check if (target ^ a) is in available for any a in available.
    # This is O(|available|) per target.
    # |available| grows as we add gates.
    
    # Let's limit the number of iterations to avoid O(N^2) blowup if it's slow.
    
    num_gates = 0
    next_wire = n_input + 1
    
    # We need to track which targets are satisfied.
    # satisfied_targets: mask -> wire_index
    
    while targets and time.time() < deadline:
        found_match = False
        
        # Sort targets by weight (popcount) to handle simpler ones first?
        # Or just iterate.
        target_list = list(targets.keys())
        
        for t_mask in target_list:
            # Check if t_mask can be formed by XOR of two available masks
            # Iterate through available masks
            # Optimization: only check a subset if available is large?
            # For now, check all.
            
            # To avoid a^b == b^a, we can just check all pairs?
            # No, we just need to find ONE pair.
            
            # We can iterate through available items.
            # If len(available) is large, this is slow.
            # Let's cap the check.
            
            for a_mask, a_wire in available.items():
                b_mask = t_mask ^ a_mask
                if b_mask in available:
                    b_wire = available[b_mask]
                    # Found a match!
                    # Create gate
                    gates = [[a_wire, b_wire]]
                    # We need to accumulate gates. Let's use a list.
                    # Wait, the return format is {'gates': [[w1, w2], ...], 'outputs': [...]}
                    # I need to maintain the list of gates.
                    
                    # Let's restructure to maintain state properly.
                    pass
        
        # The above loop structure is messy for maintaining state. Let's rewrite cleanly.
        
        break # Exit dummy loop to rewrite
        
    # --- Rewritten Clean Logic ---
    
    available = {}
    for i in range(n_input):
        available[1 << i] = i
    available[0] = n_input
    
    targets = {}
    for i in range(n_output):
        mask = current_funcs[i]
        if mask not in targets:
            targets[mask] = []
        targets[mask].append(i)
        
    for mask in list(targets.keys()):
        if mask in available:
            del targets[mask]
            
    gates = []
    next_wire = n_input + 1
    
    # Greedy CSE
    while targets and time.time() < deadline:
        matched = False
        # Try to find a target that can be formed by XOR of two availables
        # To make this efficient, we can iterate through targets and for each,
        # check if there's an available a such that t^a is also available.
        
        # We can optimize by checking only a few candidates from available?
        # Or just all if size is small.
        
        # Let's iterate through targets
        for t_mask in list(targets.keys()):
            # Check all available a
            # Optimization: if len(available) > 100, this might be slow.
            # But n_input is max 16, so max unique masks is 2^16 = 65536.
            # However, we only add masks that are targets or intermediates.
            # Number of targets is <= n_output (16).
            # Number of intermediates we add is bounded by gates.
            
            # Let's just do it.
            for a_mask, a_wire in available.items():
                b_mask = t_mask ^ a_mask
                if b_mask in available:
                    b_wire = available[b_mask]
                    # Found
                    gates.append([a_wire, b_wire])
                    available[t_mask] = next_wire
                    next_wire += 1
                    del targets[t_mask]
                    matched = True
                    break
            if matched:
                break
        
        if not matched:
            # No direct match found.
            # We need to create a new intermediate or give up and do independent synthesis.
            # Creating an intermediate is risky if it doesn't lead to savings.
            # Let's try one step of "decomposition".
            # Pick the target with the smallest weight > 1.
            # Decompose it into two parts?
            # Actually, if we can't find a match, it means no two current wires XOR to any target.
            # This implies the targets are "independent" of the current basis in a linear algebra sense?
            # No, it just means no direct 2-gate completion.
            
            # Let's try to create an intermediate that is a "part" of a target.
            # Heuristic: Pick a target T. Pick a bit i in T.
            # Create wire W = T ^ (1<<i).
            # Then T = W ^ (1<<i).
            # This adds 1 gate.
            # Is it worth it?
            # Only if W is useful for other targets.
            
            # Let's check if creating such an intermediate helps.
            # We can try a few candidates.
            
            candidate_created = False
            # Pick target with max weight? Or min weight?
            # Min weight is easier to decompose.
            sorted_targets = sorted(targets.keys(), key=lambda m: bin(m).count('1'))
            
            for t_mask in sorted_targets:
                # Try to decompose t_mask
                # We want to split t_mask into a and b such that a and b are "simpler" or "available".
                # Since no a,b in available XOR to t_mask, we must create a new one.
                # Let's pick a sub-mask of t_mask.
                # How to pick?
                # Maybe pick a single bit?
                # If we create W = t_mask ^ (1<<i), then W has weight weight(t_mask)-1.
                # This reduces the weight of the remaining part.
                
                # Let's try creating W for each bit i in t_mask.
                # Check if W is already available? (We know it's not, because if it were, t_mask would match).
                # Check if W can help other targets?
                # This is expensive to check.
                
                # Simple heuristic: Just create the intermediate for the first bit of the first target.
                # This ensures progress (reduces max weight of remaining targets).
                
                # Find first set bit in t_mask
                i = 0
                while (t_mask >> i) & 1 == 0:
                    i += 1
                bit_mask = 1 << i
                new_mask = t_mask ^ bit_mask
                
                # Create gate for new_mask
                # We need to synthesize new_mask from availables?
                # Wait, new_mask is not necessarily available.
                # We are creating a NEW wire.
                # But we need to define what the gate computes.
                # The gate computes XOR of two EXISTING wires.
                # So we can't just "create" new_mask arbitrarily.
                # We must XOR two existing wires to get new_mask.
                # This means new_mask must be representable as XOR of two availables.
                # But we just checked that NO target is representable.
                # new_mask is NOT a target (it's a sub-part).
                # So maybe new_mask IS representable by two availables?
                
                # Check if new_mask can be formed by two availables.
                formed = False
                for a_mask, a_wire in available.items():
                    b_mask = new_mask ^ a_mask
                    if b_mask in available:
                        b_wire = available[b_mask]
                        # We can form new_mask
                        gates.append([a_wire, b_wire])
                        available[new_mask] = next_wire
                        next_wire += 1
                        formed = True
                        candidate_created = True
                        break
                if formed:
                    # Now t_mask can be formed by new_mask ^ bit_mask (which is available)
                    # We don't need to add t_mask to available yet, but it will be found in next iteration.
                    # Or we can add it now?
                    # t_mask = new_mask ^ bit_mask
                    # new_mask is available, bit_mask is available.
                    # So t_mask is now formable.
                    # We can let the next iteration handle it, or handle it now.
                    # Handling it now saves an iteration.
                    # But let's just break and let the loop restart.
                    break
            
            if not candidate_created:
                # We couldn't even form a sub-part.
                # This means the available set is "closed" under XOR for these sub-parts?
                # Or we are stuck.
                # Fallback: Independent synthesis for remaining targets.
                # For each remaining target, synthesize it from scratch using inputs.
                # This is the baseline.
                
                for t_mask in list(targets.keys()):
                    # Synthesize t_mask
                    # We can use a simple tree or chain.
                    # Chain: XOR inputs one by one.
                    # This is simple and guarantees correctness.
                    # Cost: weight(t_mask) - 1 gates.
                    
                    # Find the set bits
                    bits = [i for i in range(n_input) if (t_mask >> i) & 1]
                    if not bits:
                        # Zero
                        available[t_mask] = n_input
                        del targets[t_mask]
                        continue
                    
                    if len(bits) == 1:
                        available[t_mask] = bits[0]
                        del targets[t_mask]
                        continue
                    
                    # Chain
                    current_wire = bits[0]
                    for b in bits[1:]:
                        gates.append([current_wire, b])
                        current_wire = next_wire
                        next_wire += 1
                        # Note: we don't add intermediate wires to available unless they are useful.
                        # For independent synthesis, they are not.
                    
                    available[t_mask] = current_wire
                    del targets[t_mask]
                
                break # Exit while loop
            
            # If we created a candidate, loop continues to try to match targets again.

    # Construct output
    outputs = []
    for i in range(n_output):
        mask = current_funcs[i]
        outputs.append(available[mask])
        
    return {'gates': gates, 'outputs': outputs}

# EVOLVE-BLOCK-END
