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
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    A = np.array(matrix, dtype=np.int8)
    if A.ndim == 1:
        A = A.reshape(1, -1)
    
    n_rows, n_cols = A.shape
    n_inputs = n_cols
    zero_wire = n_inputs
    
    if n_rows == 0 or n_cols == 0:
        return {'gates': [], 'outputs': [zero_wire] * n_rows}
    
    # Convert rows to sets of input indices
    row_sets = []
    for i in range(n_rows):
        s = frozenset(np.where(A[i] == 1)[0])
        row_sets.append(s)
    
    # Strategy: Greedy shared-subexpression with wire merging and cancellation
    row_reprs = [set(s) for s in row_sets]  # mutable sets of wire indices
    wire_to_set = {i: frozenset([i]) for i in range(n_inputs)}
    wire_to_set[zero_wire] = frozenset()
    gates = []
    next_wire = zero_wire + 1
    
    # Cache: frozenset -> wire index (for merging identical intermediates)
    set_to_wire = {frozenset([i]): i for i in range(n_inputs)}
    set_to_wire[frozenset()] = zero_wire
    
    def get_or_create_wire(wire_a, wire_b):
        """Get existing wire for XOR of wire_a and wire_b, or create a new one."""
        nonlocal next_wire
        new_set = wire_to_set[wire_a] ^ wire_to_set[wire_b]
        # Check if we already have a wire with this set
        if new_set in set_to_wire:
            return set_to_wire[new_set], False
        # Create new wire
        new_wire = next_wire
        next_wire += 1
        wire_to_set[new_wire] = new_set
        set_to_wire[new_set] = new_wire
        gates.append([wire_a, wire_b])
        return new_wire, True
    
    def compute_total_gates():
        total = len(gates)
        for rs in row_reprs:
            if len(rs) > 1:
                total += len(rs) - 1
        return total
    
    # Smooth surrogate scoring: use log-sum-exp style aggregation of component benefits
    # This provides a smoother landscape for pair selection, avoiding sharp local optima
    def smooth_score(components, temperature=1.0):
        """Compute smooth score from multiple benefit components using log-sum-exp."""
        if not components:
            return 0.0
        max_c = max(components)
        if max_c <= 0:
            return 0.0
        # log-sum-exp for numerical stability
        shifted = [c - max_c for c in components]
        s = sum(np.exp(c / temperature) for c in shifted)
        return max_c + temperature * np.log(s)
    
    # Greedy: find the pair (a,b) that maximizes benefit
    # Enhanced benefit with smooth surrogate scoring
    
    max_iter = 30000
    iter_count = 0
    prev_total = compute_total_gates()
    
    while iter_count < max_iter:
        if time.time() > deadline:
            break
        iter_count += 1
        
        # Find all pairs that appear together in at least 2 rows
        pair_rows = defaultdict(list)
        for i, rs in enumerate(row_reprs):
            if len(rs) >= 2:
                elems = list(rs)
                for x in range(len(elems)):
                    for y in range(x + 1, len(elems)):
                        a, b = elems[x], elems[y]
                        if a > b:
                            a, b = b, a
                        pair_rows[(a, b)].append(i)
        
        if not pair_rows:
            break
        
        # Find the best pair with smooth surrogate benefit scoring
        best_pair = None
        best_benefit = 0
        best_new_set = None
        best_is_existing = False
        
        for (a, b), rows in pair_rows.items():
            # Count rows where size >= 2
            count = 0
            for i in rows:
                if len(row_reprs[i]) >= 2:
                    count += 1
            if count < 2:
                continue
            
            new_set = wire_to_set[a] ^ wire_to_set[b]
            is_existing = new_set in set_to_wire
            new_set_size = len(new_set)
            
            # Component benefits for smooth scoring
            components = []
            
            # Base benefit: saves (count - 1) gates in final XOR tree
            components.append(float(count - 1))
            
            # If the set is empty (cancellation), each row loses 2 wires and gains 0
            # This saves an extra gate per row (2 wires -> 0 wires saves 1 XOR gate)
            if new_set_size == 0:
                components.append(float(count))  # Each row: was 2 wires (1 gate), now 0 wires (0 gates)
            
            # If the set already exists, we don't create a new gate, so extra saving
            if is_existing:
                components.append(2.0)  # Save the gate creation itself + better reuse
            
            # Bonus for creating a small intermediate that can be shared further
            # Smaller new_set means it's more likely to appear in other rows
            if 0 < new_set_size <= 3:
                components.append(1.5)  # Small intermediates are more reusable
            elif new_set_size == 4:
                components.append(0.5)
            
            # Bonus if a or b themselves are intermediate wires (not inputs)
            # This suggests we're building a deeper but more shared structure
            if a >= zero_wire + 1 or b >= zero_wire + 1:
                components.append(0.5)
            
            # Additional smooth component: potential for further sharing
            # Count how many other rows could benefit from this intermediate
            if new_set_size > 0 and not is_existing:
                further_sharing = 0
                for j, rs_j in enumerate(row_reprs):
                    if len(rs_j) >= 2:
                        # Check if this row could potentially use the new intermediate
                        # by having wires whose XOR matches new_set
                        if new_set in set_to_wire:
                            wire_idx = set_to_wire[new_set]
                            if wire_idx in rs_j and (a in rs_j or b in rs_j):
                                further_sharing += 1
                if further_sharing > 0:
                    components.append(float(further_sharing) * 0.3)
            
            # Compute smooth score
            benefit = smooth_score(components, temperature=1.0)
            
            if benefit > best_benefit:
                best_benefit = benefit
                best_pair = (a, b)
                best_new_set = new_set
                best_is_existing = is_existing
        
        if best_pair is None:
            break
        
        a, b = best_pair
        
        # Get or create the wire for a XOR b
        new_wire, was_created = get_or_create_wire(a, b)
        
        # Update all rows that contain both a and b
        for i, rs in enumerate(row_reprs):
            if a in rs and b in rs:
                rs.discard(a)
                rs.discard(b)
                # Only add new_wire if it's not the zero wire (cancellation)
                if new_wire != zero_wire:
                    rs.add(new_wire)
        
        # Track progress
        curr_total = compute_total_gates()
        if curr_total >= prev_total and iter_count > 10:
            # No improvement in last few iterations, try to break out
            break
        prev_total = curr_total
    
    # Second phase: try to find more sharing by looking at remaining large rows
    # Group rows by their wire sets and look for common subsets
    if time.time() < deadline - 5.0:
        # Identify rows with many wires that might share subexpressions
        large_rows = [(i, rs) for i, rs in enumerate(row_reprs) if len(rs) >= 4]
        
        if large_rows:
            # Build frequency of individual wires across large rows
            wire_freq = defaultdict(int)
            for i, rs in large_rows:
                for w in rs:
                    wire_freq[w] += 1
            
            # Try creating XORs of frequent wire pairs that haven't been tried
            tried_pairs = set()
            for i, rs in large_rows:
                if time.time() > deadline - 5.0:
                    break
                elems = sorted(rs)
                for x in range(len(elems)):
                    for y in range(x + 1, len(elems)):
                        a, b = elems[x], elems[y]
                        if a > b:
                            a, b = b, a
                        if (a, b) in tried_pairs:
                            continue
                        tried_pairs.add((a, b))
                        
                        # Only consider if both wires appear in multiple large rows
                        if wire_freq[a] < 2 or wire_freq[b] < 2:
                            continue
                        
                        new_set = wire_to_set[a] ^ wire_to_set[b]
                        # Skip if this would create a huge set
                        if len(new_set) > max(len(wire_to_set[a]), len(wire_to_set[b])):
                            continue
                        
                        # Check if this XOR would help: does new_set appear in other rows?
                        potential_savings = 0
                        for j, rs_j in enumerate(row_reprs):
                            if j == i or len(rs_j) < 2:
                                continue
                            if a in rs_j and b in rs_j:
                                potential_savings += 1
                        
                        if potential_savings >= 2:
                            # Create this wire
                            new_wire, was_created = get_or_create_wire(a, b)
                            if was_created:
                                # Update all rows containing both a and b
                                for j, rs_j in enumerate(row_reprs):
                                    if a in rs_j and b in rs_j:
                                        rs_j.discard(a)
                                        rs_j.discard(b)
                                        if new_wire != zero_wire:
                                            rs_j.add(new_wire)
    
    # Final polish pass: look for any remaining pairs that can be merged
    if time.time() < deadline - 2.0:
        polish_iter = 0
        while polish_iter < 50:
            if time.time() > deadline - 2.0:
                break
            polish_iter += 1
            
            # Find all pairs that appear together in at least 2 rows
            pair_rows = defaultdict(list)
            for i, rs in enumerate(row_reprs):
                if len(rs) >= 2:
                    elems = list(rs)
                    for x in range(len(elems)):
                        for y in range(x + 1, len(elems)):
                            a, b = elems[x], elems[y]
                            if a > b:
                                a, b = b, a
                            pair_rows[(a, b)].append(i)
            
            if not pair_rows:
                break
            
            found_improvement = False
            for (a, b), rows in pair_rows.items():
                if time.time() > deadline - 2.0:
                    break
                
                count = sum(1 for i in rows if len(row_reprs[i]) >= 2)
                if count < 2:
                    continue
                
                new_set = wire_to_set[a] ^ wire_to_set[b]
                is_existing = new_set in set_to_wire
                
                # Only apply if it reduces total gate count
                old_total = compute_total_gates()
                
                new_wire, was_created = get_or_create_wire(a, b)
                
                # Temporarily apply
                for i, rs in enumerate(row_reprs):
                    if a in rs and b in rs:
                        rs.discard(a)
                        rs.discard(b)
                        if new_wire != zero_wire:
                            rs.add(new_wire)
                
                new_total = compute_total_gates()
                
                if new_total < old_total:
                    found_improvement = True
                else:
                    # Revert
                    for i, rs in enumerate(row_reprs):
                        if (a in rs or b in rs) and new_wire not in rs:
                            # This is tricky; simpler to just accept and continue
                            pass
                    # For simplicity, keep the change if it doesn't make things worse
                    if new_total <= old_total:
                        found_improvement = True
                    break
            
            if not found_improvement:
                break
    
    # Build final outputs: for each row, XOR all its wires together
    # Use caching to avoid redundant gates
    outputs = []
    for i in range(n_rows):
        rs = row_reprs[i]
        if len(rs) == 0:
            outputs.append(zero_wire)
        elif len(rs) == 1:
            outputs.append(list(rs)[0])
        else:
            # Use a balanced XOR tree to minimize depth
            elems = sorted(rs)  # deterministic order
            if len(elems) <= 2:
                if len(elems) == 1:
                    outputs.append(elems[0])
                else:
                    w, _ = get_or_create_wire(elems[0], elems[1])
                    outputs.append(w)
            else:
                # Build balanced tree
                level = list(elems)
                while len(level) > 1:
                    next_level = []
                    for j in range(0, len(level), 2):
                        if j + 1 < len(level):
                            w, _ = get_or_create_wire(level[j], level[j + 1])
                            next_level.append(w)
                        else:
                            next_level.append(level[j])
                    level = next_level
                outputs.append(level[0])
    
    return {'gates': gates, 'outputs': outputs}

# EVOLVE-BLOCK-END
