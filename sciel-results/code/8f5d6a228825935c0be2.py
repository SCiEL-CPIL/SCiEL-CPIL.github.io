"""XOR-circuit synthesis using greedy pairing to minimize intermediate weights."""

def synthesize(matrix, depth_limit):
    import numpy as np
    from collections import defaultdict
    
    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0
    
    if m == 0:
        return {'gates': [], 'outputs': []}
    
    # Convert to list of tuples
    rows = [tuple(matrix[i]) for i in range(m)]
    
    # Wire mapping
    # 0..n-1: inputs
    # n: zero
    # n+1...: gates
    
    gates = []
    wire_to_vec = {}
    vec_to_wire = {}
    
    # Initialize inputs
    for i in range(n):
        vec = [0] * n
        vec[i] = 1
        v = tuple(vec)
        wire_to_vec[i] = v
        vec_to_wire[v] = i
    
    # Initialize zero
    zero_vec = tuple([0] * n)
    wire_to_vec[n] = zero_vec
    vec_to_wire[zero_vec] = n
    
    next_wire = n + 1
    
    def xor_vec(a, b):
        return tuple(x ^ y for x, y in zip(a, b))
    
    def add_gate(w1, w2):
        nonlocal next_wire
        v1 = wire_to_vec[w1]
        v2 = wire_to_vec[w2]
        new_v = xor_vec(v1, v2)
        
        # Check if we already have this vector
        if new_v in vec_to_wire:
            return vec_to_wire[new_v]
        
        gates.append([w1, w2])
        wire_to_vec[next_wire] = new_v
        vec_to_wire[new_v] = next_wire
        new_wire = next_wire
        next_wire += 1
        return new_wire

    # Targets to compute
    # Each target is a vector tuple
    targets = list(rows)
    
    # We will iteratively simplify the targets
    # We want to reduce the sum of weights of the targets
    # By replacing t1, t2 with u=t1^t2, t2
    
    # To avoid infinite loops, we limit iterations or ensure weight decreases
    max_iterations = 100
    for _ in range(max_iterations):
        best_pair_idx = -1
        best_gain = 0
        best_u = None
        best_t1_idx = -1
        best_t2_idx = -1
        
        # Find the best pair to merge
        # Gain = wt(t1) - wt(u)  (since t2 is kept, total weight change is wt(u) - wt(t1))
        # We want to minimize total weight, so we want max gain (wt(t1) - wt(u) > 0)
        
        for i in range(len(targets)):
            t1 = targets[i]
            wt1 = sum(t1)
            for j in range(i + 1, len(targets)):
                t2 = targets[j]
                u = xor_vec(t1, t2)
                wt_u = sum(u)
                
                # If we replace t1 with u, the set becomes {u, t2, ...}
                # New total weight = Old - wt1 + wt_u
                # Gain = wt1 - wt_u
                if wt_u < wt1:
                    gain = wt1 - wt_u
                    if gain > best_gain:
                        best_gain = gain
                        best_pair_idx = i
                        best_t1_idx = i
                        best_t2_idx = j
                        best_u = u
        
        if best_gain <= 0:
            break
            
        # Perform the merge
        # We replace targets[best_t1_idx] with best_u
        # But we must remember that the original target was targets[best_t1_idx]
        # And it is now computed as best_u ^ targets[best_t2_idx]
        
        # We need to track dependencies.
        # Let's use a dictionary mapping target_vector to (wire_or_target, dependency)
        # Actually, since vectors are unique, we can just track the transformation.
        
        # This simple replacement loses the information of WHICH target was which.
        # We need to keep the original targets for output.
        
        # Let's restart the strategy.
        # We build a dependency tree.
        # We maintain a list of "current nodes" in the circuit.
        # Initially, nodes = [0, ..., n-1, n] (wires)
        # We want to generate the output vectors.
        
        # This is getting messy. Let's use the DP/Matching idea on the outputs directly.
        # Given the small m, we can try to find a minimum weight basis for the outputs.
        
        # Let's go back to the simple Greedy Compute with Caching, but optimize the order.
        # Sort outputs by weight (ascending).
        # Compute lighter outputs first. They are likely to be intermediates for heavier ones.
        
        pass # Placeholder for logic below

    # Re-implementing with the "Sort by Weight" and "Cache" approach, which is robust.
    # The previous code did this but maybe not optimally.
    
    # Let's use the "Pairing" idea properly by building a tree.
    
    # We will create a list of "jobs" to be done.
    # Each job is a vector.
    # We start with the output vectors.
    # We repeatedly pick a job and split it into two simpler jobs if possible.
    # Split t into t1, t2 such that t = t1 ^ t2.
    # We prefer t1, t2 with lower weights.
    # If t1 or t2 is already available (in wire_to_vec), we only need the other.
    
    # This is a recursive decomposition.
    
    # Let's use a Memoized Recursion for the "Cost to Compute".
    # But we need the actual circuit.
    
    # Let's stick to the Greedy Pairing on the Set of Targets.
    # We will maintain a list of `targets`.
    # And a dict `replacements` mapping original_output_index to a list of (wire, operation) steps?
    
    # Simpler:
    # 1. Sort outputs by weight.
    # 2. For each output, try to express it using existing wires.
    # 3. If not, find the best split v = a ^ b where a, b are "simple".
    #    "Simple" means low weight or already available.
    
    # Let's implement the "Split" strategy.
    
    def get_cost(vec):
        """Estimate cost to compute vec from inputs."""
        if vec in vec_to_wire:
            return 0
        # Heuristic: Hamming weight
        return sum(vec) - 1

    # We will process outputs one by one.
    # To maximize sharing, we process them in an order that allows intermediates to be reused.
    # Sort by weight ascending.
    
    sorted_outputs = sorted([(sum(r), i, r) for i, r in enumerate(rows)], key=lambda x: x[0])
    
    outputs = [0] * m
    
    for _, idx, row in sorted_outputs:
        vec = row
        if vec in vec_to_wire:
            outputs[idx] = vec_to_wire[vec]
            continue
            
        # Find best split vec = a ^ b
        # a, b should be such that cost(a) + cost(b) is minimized.
        # We iterate over all possible a? No, too slow.
        # We iterate over available wires a.
        # b = vec ^ a.
        # Cost = 1 + cost(b) (if b is not available) or 1 (if b is available).
        # Wait, if we compute b, we pay for b.
        
        best_cost = float('inf')
        best_a = None
        best_b = None
        best_wire_a = None
        best_wire_b = None
        
        # Try all available wires as a
        for w_a, v_a in wire_to_vec.items():
            v_b = xor_vec(vec, v_a)
            # Check if v_b is available
            if v_b in vec_to_wire:
                cost = 1 # Just the gate for vec = a ^ b
                # Prefer lower index or something?
                if cost < best_cost:
                    best_cost = cost
                    best_a = v_a
                    best_b = v_b
                    best_wire_a = w_a
                    best_wire_b = vec_to_wire[v_b]
            else:
                # Estimate cost of v_b
                # If we compute v_b, we need to generate it.
                # cost(v_b) is roughly weight(v_b) - 1.
                # But if we generate v_b, it might be useful later.
                # For now, just use weight.
                est_cost_b = sum(v_b) - 1
                total_cost = 1 + est_cost_b
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_a = v_a
                    best_b = v_b
                    best_wire_a = w_a
                    best_wire_b = None # Need to generate
        
        if best_wire_b is None:
            # Generate best_b first
            # Recursively? Or just greedily generate it now.
            # To avoid deep recursion, we can just generate it using the same logic.
            # But we must be careful not to create infinite loops.
            
            # Let's just generate best_b using a simple greedy split of best_b.
            # This is inefficient but works for small n.
            
            # Actually, let's just compute best_b directly from inputs if it's not in the "available" set?
            # No, that defeats the purpose.
            
            # Let's use a helper to compute any vector.
            def compute_vec(v):
                if v in vec_to_wire:
                    return vec_to_wire[v]
                
                # Find best split
                min_c = float('inf')
                ba = None
                bb = None
                wa = None
                wb = None
                for w_a, v_a in wire_to_vec.items():
                    v_b = xor_vec(v, v_a)
                    if v_b in vec_to_wire:
                        c = 1
                        if c < min_c:
                            min_c = c
                            ba = v_a
                            bb = v_b
                            wa = w_a
                            wb = vec_to_wire[v_b]
                    else:
                        # Don't recurse deep, just estimate
                        c = 1 + sum(v_b)
                        if c < min_c:
                            min_c = c
                            ba = v_a
                            bb = v_b
                            wa = w_a
                            wb = None
                            
                if wb is None:
                    # Compute bb
                    # To prevent infinite recursion, we can fall back to direct computation
                    # Direct computation:
                    idxs = [i for i in range(n) if bb[i] == 1]
                    if not idxs:
                        wb = n
                    else:
                        wb = idxs[0]
                        for idx in idxs[1:]:
                            wb = add_gate(wb, idx)
                else:
                    # We have wa and wb
                    # Compute v = wa ^ wb
                    pass # Handled by caller? No, this is inside compute_vec.
                
                # This is getting messy.
                
                # Let's restart with a cleaner "BFS with Sharing"
                pass
            
            # Let's use the BFS approach from the seed but improved.
            # The seed's BFS was O(N^2) per step.
            
            # Let's just use the "Sort by Weight" + "Direct Compute with Cache" from the seed, 
            # but ensure we check for existing wires first.
            
            # The seed code actually did this.
            # Why was the score 1.09?
            # Maybe the "Direct Compute" fallback was too expensive.
            
            # Let's try the Pairing again, but simpler.
            # Just sort outputs by weight.
            # For each output, if not in cache, compute it by XORing with the "closest" available wire.
            
            # We'll implement a helper `get_wire(vec)` that creates the wire if it doesn't exist.
            # To create it, it picks the available wire `a` that minimizes `weight(vec ^ a)`.
            # Then it recursively gets `vec ^ a`.
            # To avoid infinite recursion, we limit depth or use memoization.
            
            pass

    # Final implementation:
    # 1. Sort outputs by weight.
    # 2. For each output, call `get_wire(row)`.
    # 3. `get_wire` checks cache.
    # 4. If not in cache, finds `a` in cache minimizing `weight(row ^ a)`.
    # 5. Recursively calls `get_wire(row ^ a)`.
    # 6. Creates gate `row = a ^ (row ^ a)`.
    # 7. Returns wire.
    
    # This is a standard heuristic.
    
    # Let's rewrite the function body completely to be clean.
    
    # (See actual code block below)

    # Re-initialize state for the new logic
    gates = []
    wire_to_vec = {}
    vec_to_wire = {}
    
    for i in range(n):
        vec = [0] * n
        vec[i] = 1
        v = tuple(vec)
        wire_to_vec[i] = v
        vec_to_wire[v] = i
        
    zero_vec = tuple([0] * n)
    wire_to_vec[n] = zero_vec
    vec_to_wire[zero_vec] = n
    
    next_wire = n + 1
    
    def xor_vec(a, b):
        return tuple(x ^ y for x, y in zip(a, b))
        
    def add_gate(w1, w2):
        nonlocal next_wire
        v1 = wire_to_vec[w1]
        v2 = wire_to_vec[w2]
        new_v = xor_vec(v1, v2)
        if new_v in vec_to_wire:
            return vec_to_wire[new_v]
        gates.append([w1, w2])
        wire_to_vec[next_wire] = new_v
        vec_to_wire[new_v] = next_wire
        new_wire = next_wire
        next_wire += 1
        return new_wire

    import sys
    sys.setrecursionlimit(10000)

    def get_wire(vec):
        vec = tuple(vec)
        if vec in vec_to_wire:
            return vec_to_wire[vec]
        
        # Find best a in wire_to_vec
        best_a = None
        best_diff = None
        min_wt = float('inf')
        
        for w, v in wire_to_vec.items():
            diff = xor_vec(vec, v)
            wt = sum(diff)
            if wt < min_wt:
                min_wt = wt
                best_a = w
                best_diff = diff
                
        if best_diff is None:
            # Fallback
            return n
            
        # Recursively get best_diff
        # To prevent infinite recursion if best_diff is "hard", 
        # we check if best_diff is simpler (lower weight) than vec.
        # If not, we might loop.
        # But since we pick min weight diff, and vec is fixed, 
        # if min_wt >= weight(vec), we should probably just compute directly.
        
        wt_vec = sum(vec)
        if min_wt >= wt_vec:
            # Direct computation
            idxs = [i for i in range(n) if vec[i] == 1]
            if not idxs:
                return n
            w_res = idxs[0]
            for idx in idxs[1:]:
                w_res = add_gate(w_res, idx)
            return w_res
            
        w_diff = get_wire(best_diff)
        w_res = add_gate(best_a, w_diff)
        return w_res

    outputs = [0] * m
    # Sort by weight to encourage sharing of simple intermediates
    sorted_indices = sorted(range(m), key=lambda i: sum(rows[i]))
    
    for idx in sorted_indices:
        outputs[idx] = get_wire(rows[idx])
        
    return {'gates': gates, 'outputs': outputs}
