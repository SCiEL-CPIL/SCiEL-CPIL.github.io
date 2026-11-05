"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import defaultdict, deque
import heapq

def synthesize(matrix, depth_limit):
    """Synthesize an XOR circuit using iterative subexpression extraction with dynamic benefit recalculation.
    
    Uses a priority queue to dynamically re-evaluate extraction benefits as the circuit evolves,
    enabling deeper sharing opportunities that single-pass approaches miss.
    """
    start_time = time.time()
    DEADLINE = 82.0  # Safety margin
    
    mat = np.array(matrix, dtype=np.int8)
    n_outputs, n_inputs = mat.shape
    
    def row_to_mask(row):
        m = 0
        for j, v in enumerate(row):
            if v:
                m |= (1 << j)
        return m
    
    output_masks = [row_to_mask(mat[i]) for i in range(n_outputs)]
    
    def popcount(m):
        return bin(m).count('1')
    
    def bits_from_mask(m):
        bits = []
        j = 0
        while m:
            if m & 1:
                bits.append(j)
            m >>= 1
            j += 1
        return bits
    
    # State tracking
    expr_to_wire = {}
    for i in range(n_inputs):
        expr_to_wire[(1 << i)] = i
    expr_to_wire[0] = n_inputs
    
    next_wire = n_inputs + 1
    gates = []
    
    # Track which outputs use which expressions
    remaining_masks = list(output_masks)
    output_exprs = [[] for _ in range(n_outputs)]
    
    def compute_expression_depth(expr):
        """Compute the depth needed to synthesize this expression from inputs."""
        if expr in expr_to_wire:
            # Check if it's an input
            if expr == 0:
                return 0
            if expr & (expr - 1) == 0:  # power of 2
                return 0
            # It's a previously created gate - we'd need to track depth, approximate as 1
            return 1
        bits = bits_from_mask(expr)
        if len(bits) <= 1:
            return 0
        # Balanced tree depth
        d = 0
        n = len(bits)
        while n > 1:
            d += 1
            n = (n + 1) // 2
        return d
    
    def create_expression_wires(expr, bits):
        """Create wires for the expression, return the final wire."""
        nonlocal next_wire
        k = len(bits)
        
        if k == 1:
            return bits[0]
        
        # Build balanced tree
        level = list(bits)
        while len(level) > 1:
            new_level = []
            for i in range(0, len(level) - 1, 2):
                gates.append([level[i], level[i+1]])
                new_level.append(next_wire)
                next_wire += 1
            if len(level) % 2 == 1:
                new_level.append(level[-1])
            level = new_level
        return level[0]
    
    def extract_expression(expr, bits):
        """Extract an expression and update state. Returns the wire created."""
        nonlocal next_wire
        w = create_expression_wires(expr, bits)
        expr_to_wire[expr] = w
        
        for i in range(n_outputs):
            if (remaining_masks[i] & expr) == expr:
                remaining_masks[i] ^= expr
                output_exprs[i].append(w)
        
        return w
    
    # Priority queue: (-benefit, expr, frozenset(bits))
    # We'll use a counter to break ties
    pq = []
    pq_counter = 0
    
    def try_extract(expr):
        """Try to extract an expression if beneficial."""
        nonlocal pq_counter
        bits = bits_from_mask(expr)
        k = len(bits)
        
        if k < 2:
            return False
        
        if expr in expr_to_wire:
            return False
        
        # Count how many outputs still contain this expression
        count = sum(1 for i in range(n_outputs) if (remaining_masks[i] & expr) == expr)
        
        if count < 2:
            return False
        
        # Benefit: gates saved = (k-1) gates per use, but we pay (k-1) to create it
        # Net benefit = (k-1) * (count - 1)
        benefit = (k - 1) * (count - 1)
        
        if benefit <= 0:
            return False
        
        heapq.heappush(pq, (-benefit, pq_counter, expr, frozenset(bits)))
        pq_counter += 1
        return True
    
    # Initial candidate generation: all pairs, triples, quadruples from each output
    # But we'll be more systematic - generate all subsets of size 2-5 from each output
    # and deduplicate
    
    seen_exprs = set()
    
    for i in range(n_outputs):
        bits = bits_from_mask(output_masks[i])
        n = len(bits)
        
        # Generate all subsets of size 2 to min(5, n)
        max_size = min(5, n)
        
        # Size 2
        for a in range(n):
            for b in range(a+1, n):
                expr = (1 << bits[a]) | (1 << bits[b])
                if expr not in seen_exprs:
                    seen_exprs.add(expr)
                    try_extract(expr)
        
        # Size 3
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    expr = (1 << bits[a]) | (1 << bits[b]) | (1 << bits[c])
                    if expr not in seen_exprs:
                        seen_exprs.add(expr)
                        try_extract(expr)
        
        # Size 4
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    for d in range(c+1, n):
                        expr = (1 << bits[a]) | (1 << bits[b]) | (1 << bits[c]) | (1 << bits[d])
                        if expr not in seen_exprs:
                            seen_exprs.add(expr)
                            try_extract(expr)
        
        # Size 5 (only if output has >= 5 bits)
        if n >= 5:
            for a in range(n):
                for b in range(a+1, n):
                    for c in range(b+1, n):
                        for d in range(c+1, n):
                            for e in range(d+1, n):
                                expr = (1 << bits[a]) | (1 << bits[b]) | (1 << bits[c]) | (1 << bits[d]) | (1 << bits[e])
                                if expr not in seen_exprs:
                                    seen_exprs.add(expr)
                                    try_extract(expr)
    
    # Iterative extraction with dynamic re-evaluation
    max_iterations = 500
    iter_count = 0
    
    while pq and iter_count < max_iterations:
        if time.time() > DEADLINE:
            break
        
        neg_benefit, _, expr, bits_frozen = heapq.heappop(pq)
        benefit = -neg_benefit
        
        # Skip if already extracted
        if expr in expr_to_wire:
            continue
        
        # Re-validate: check if still beneficial
        bits = list(bits_frozen)
        k = len(bits)
        
        count = sum(1 for i in range(n_outputs) if (remaining_masks[i] & expr) == expr)
        if count < 2:
            continue
        
        current_benefit = (k - 1) * (count - 1)
        if current_benefit <= 0:
            continue
        
        # Extract this expression
        extract_expression(expr, bits)
        
        # After extraction, re-evaluate other expressions that might now be more/less beneficial
        # Also, new combinations might become extractable
        # For efficiency, only re-check expressions that share bits with the extracted one
        extracted_bits = set(bits)
        
        # Re-check all expressions in the priority queue that share bits
        new_pq = []
        rechecked = set()
        
        while pq:
            neg_b, cnt, e, bf = heapq.heappop(pq)
            e_bits = set(bf)
            
            if e_bits & extracted_bits:
                # This expression shares bits with what we just extracted
                # Re-evaluate its benefit
                if e not in expr_to_wire:
                    e_count = sum(1 for i in range(n_outputs) if (remaining_masks[i] & e) == e)
                    if e_count >= 2:
                        e_k = len(e_bits)
                        e_benefit = (e_k - 1) * (e_count - 1)
                        if e_benefit > 0:
                            heapq.heappush(new_pq, (-e_benefit, cnt, e, bf))
                            rechecked.add(e)
            else:
                heapq.heappush(new_pq, (neg_b, cnt, e, bf))
        
        pq = new_pq
        
        # Also, check if the remaining parts of outputs form new extractable expressions
        # This is expensive, so only do it periodically
        if iter_count % 10 == 0:
            seen_exprs.clear()
            for i in range(n_outputs):
                if remaining_masks[i] == 0:
                    continue
                bits_rem = bits_from_mask(remaining_masks[i])
                n_rem = len(bits_rem)
                
                # Check pairs in remaining
                for a in range(n_rem):
                    for b in range(a+1, n_rem):
                        expr2 = (1 << bits_rem[a]) | (1 << bits_rem[b])
                        if expr2 not in seen_exprs and expr2 not in expr_to_wire:
                            seen_exprs.add(expr2)
                            try_extract(expr2)
                
                # Check triples in remaining
                for a in range(n_rem):
                    for b in range(a+1, n_rem):
                        for c in range(b+1, n_rem):
                            expr2 = (1 << bits_rem[a]) | (1 << bits_rem[b]) | (1 << bits_rem[c])
                            if expr2 not in seen_exprs and expr2 not in expr_to_wire:
                                seen_exprs.add(expr2)
                                try_extract(expr2)
        
        iter_count += 1
    
    # Assemble outputs with balanced trees
    output_wires = []
    for i in range(n_outputs):
        wires = list(output_exprs[i])
        rem = remaining_masks[i]
        
        # Add remaining input wires
        m = rem
        j = 0
        while m:
            if m & 1:
                wires.append(j)
            m >>= 1
            j += 1
        
        if not wires:
            output_wires.append(n_inputs)
        elif len(wires) == 1:
            output_wires.append(wires[0])
        else:
            # Balanced tree XOR
            level = wires
            while len(level) > 1:
                new_level = []
                for k in range(0, len(level) - 1, 2):
                    gates.append([level[k], level[k+1]])
                    new_level.append(next_wire)
                    next_wire += 1
                if len(level) % 2 == 1:
                    new_level.append(level[-1])
                level = new_level
            output_wires.append(level[0])
    
    return {'gates': gates, 'outputs': output_wires}

# EVOLVE-BLOCK-END
