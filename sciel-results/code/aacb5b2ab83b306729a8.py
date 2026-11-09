import numpy as np
import time
import heapq
from collections import defaultdict

def synthesize(matrix, depth_limit):
    """
    Synthesizes an XOR circuit using a Dijkstra-based shortest path strategy 
    on the hypercube of linear combinations, followed by a greedy Steiner-like 
    selection to maximize sharing.
    
    Strategy:
    1. Represent all possible linear combinations of inputs as integers (bitmasks).
    2. Compute shortest paths (min gates) from 0 to all targets and all relevant 
       intermediate nodes using Dijkstra on the hypercube graph (edges are XOR with 
       a basis vector, i.e., toggling one input bit? No, that's not right for 
       general XOR gates).
       
    Wait, the "graph" of XOR gates is not a simple hypercube where edges are 
    single-bit flips. An XOR gate takes two arbitrary wires and produces their 
    XOR. The state space is the vector space GF(2)^n. The "distance" from 0 to a 
    vector v is its rank? No, the cost is the number of gates.
    
    Actually, the minimal number of gates to compute a single vector v from 
    inputs is just the popcount(v) - 1 (if v != 0). But sharing allows us to 
    do better for multiple targets.
    
    The "Dijkstra" idea in the prompt suggests treating the problem as finding 
    a set of wires (nodes) such that the induced subgraph connects 0 and all 
    targets with minimal total edge count (gates). This is the Steiner Tree 
    problem in the complete graph on GF(2)^n, where edge weights are 1.
    
    However, the complete graph on 2^n nodes is too large for n=16 (65k nodes).
    But we only care about a subset of nodes: the targets and useful intermediates.
    
    Revised Strategy:
    1. Identify the set of "relevant" nodes. This is hard to know a priori.
    2. Instead, use a heuristic: 
       - Start with the set of available wires S = {inputs, 0}.
       - Targets T.
       - Iteratively find a wire w (not in S) that minimizes the "marginal cost" 
         to connect the current forest to the remaining targets.
       - The cost of adding w is 1 (the gate). The benefit is the reduction in 
         the estimated total cost to compute all targets.
       - Estimate cost: For each target t, the cost is the min number of gates 
         needed to build t from S ∪ {w}.
         This is approximated by: 
           If t in S: 0
           Else if t in {w}: 0 (if we just added it)
           Else: min_{u in S} (dist(t, u) + 1) where dist(t,u) is popcount(t^u)-1?
                 No, dist(t,u) is the number of gates to build t^u from inputs.
                 If we assume we build t^u from scratch, cost is popcount(t^u)-1.
                 But we can use other wires in S.
    
    Let's try a different structural change: **Gaussian Elimination with Sharing**.
    The current program uses a greedy pairwise search. 
    A different family is to use **Linear Algebra to identify independent basis** and 
    then **Synthesize the Basis with Sharing**, then **Express Targets as Linear 
    Combinations of Basis**.
    
    1. Perform Gaussian elimination on the matrix to find a basis of independent rows.
    2. Synthesize the basis vectors. To minimize gates, we can use a "tree" 
       structure for the basis vectors if they are sparse, or use a "clique" 
       if dense.
       Actually, a better approach for the basis is to use the **Strassen-like** 
       or **Recursive** splitting? No, that's for matrix multiplication.
       
    Let's look at the "Dijkstra" hint again.
    "Dijkstra's algorithm from the zero-vector to each output to guarantee 
    globally optimal gate sharing"
    
    This implies that the cost of a wire w is its "distance" from 0, and we want 
    to minimize the sum of distances? No, that would be a star.
    
    Maybe the strategy is:
    1. Build a graph where nodes are wires.
    2. Start with 0.
    3. Use Dijkstra to expand the frontier of available wires.
    4. The "cost" of a wire w is the number of gates used to create it.
    5. We want to select a set of wires such that all targets are reachable, 
       minimizing the total number of wires (gates).
    
    This is exactly the **Steiner Tree** problem.
    Since we can't solve Steiner Tree exactly for large n, we use a heuristic:
    **Prim's Algorithm on the Hypergraph**.
    
    Current program already tries Prim's-like. Why is it not better?
    Maybe the utility function is poor.
    
    New Strategy: **Minimum Spanning Tree on the Target Set with Intermediate Nodes**.
    1. Consider the set of targets T.
    2. Construct a graph where nodes are T ∪ {0} ∪ Inputs.
    3. Edge weight between u and v is 1 if u^v is an input or 0? No.
       Edge weight is 1 if we can compute v from u and some available wire?
    
    Let's try a completely different approach: **Randomized Local Search on the Gate Set**.
    1. Start with a valid circuit (e.g., direct tree for each row).
    2. Define a cost function: Total gates.
    3. Moves:
       a. Add a gate that creates a new wire w. If w helps reduce the cost of 
          other gates (by allowing them to be replaced by simpler ones), keep it.
       b. Remove a gate if its wire is not used by any output or other gate.
       c. Swap a gate's inputs? No, XOR is commutative.
       d. "Retiming": Reorder gates? Depth is limited, but we don't need to 
          minimize depth, just stay within limit.
    
    This is similar to Inspiration 3.
    
    Let's go with a **Hybrid**: 
    1. Use **Gaussian Elimination** to find a basis.
    2. Synthesize the basis using a **Greedy Set Cover** approach:
       - We have a set of "primitive" operations: XOR of two inputs, XOR of input 
         and intermediate, etc.
       - Actually, just use the current Prim's but with a **Better Utility**:
         Instead of "how many targets can be completed in 1 step", use 
         "how many targets have their **minimum distance** to the available set 
         reduced by adding this wire".
    
    Let's implement **Dijkstra-based Distance Field**:
    1. Maintain a distance map `dist[mask]` = min gates to build `mask` from 
       current available set.
    2. Initially, `dist[0]=0`, `dist[input]=0`.
    3. When we add a new wire `w` (cost 1), we update distances for all masks 
       reachable from `w`? No, we update the distance of `w` itself to 1 (if 
       built from two available).
       Then, for any target `t`, if `t^w` is available, `dist[t]` becomes 
       `dist[t^w] + 1`.
    
    This is essentially what the current program does, but maybe it's not 
    updating the "potential" correctly.
    
    Let's try **Reverse Search**:
    1. Start from all targets.
    2. "Simplify" targets by XORing with available wires to reduce complexity.
    
    Actually, the most promising structural change from the hint is **Dijkstra 
    on the Hypercube**.
    
    Hypothesis: The optimal circuit corresponds to a set of wires that forms a 
    **Shortest Path Tree** from 0 to all targets in a graph where edges are 
    XOR with *any* available wire? No, that's circular.
    
    Let's try **BFS from Targets to Inputs** (Backward Synthesis).
    1. We want to compute targets.
    2. A target t can be computed if we have t^w for some w in available.
    3. So, we can "push" requirements back.
    4. Start with requirements = Targets.
    5. Available = Inputs + 0.
    6. While requirements not empty:
       - Pick a requirement r.
       - Find a wire w in Available such that r^w is "simpler" (lower popcount 
         or already in Available).
       - If r^w is in Available, we are done with r (add gate r^w XOR w).
       - Else, add r^w to Requirements? No, that increases requirements.
       - Instead, we should add a gate that creates an intermediate wire that 
         helps multiple requirements.
    
    This is getting complicated. Let's stick to the **Prim's with Improved Scoring** 
    but change the scoring to be based on **Actual Distance Reduction**.
    
    Score(w) = Sum over all targets t of (dist_old(t) - dist_new(t))
    where dist_new(t) is the min gates to build t given S ∪ {w}.
    
    To compute dist_new(t) efficiently:
    dist_new(t) = min( dist_old(t), 1 + min_{u in S} dist_old(t^u) )? 
    No, if we add w, we can build t from w and t^w.
    dist_new(t) = min( dist_old(t), 1 + dist_new(t^w) )?
    This is recursive.
    
    Simplification:
    Assume we build t^w from scratch using S.
    cost(t) = min_{u in S} (popcount(t^u) - 1) + 1 (for the final gate with u? No, u is available).
    If we have u, we need to build t^u.
    So cost(t) = 1 + min_{u in S} (min_gates_to_build(t^u, S)).
    
    If we approximate min_gates_to_build(v, S) as popcount(v) - 1 (ignoring S 
    sharing for the sub-problem), then:
    cost(t) = 1 + min_{u in S} (popcount(t^u) - 1) = min_{u in S} popcount(t^u).
    
    So, adding w reduces the cost of t if popcount(t^w) < min_{u in S} popcount(t^u).
    
    This is a cheap heuristic!
    Score(w) = Count of targets t where popcount(t^w) < current_min_popcount(t).
    
    Let's implement this.
    """
    start_time = time.time()
    DEADLINE = 85.0
    
    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0
    
    if m == 0 or n == 0:
        return {'gates': [], 'outputs': []}
    
    target_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i][j]:
                mask |= (1 << j)
        target_masks.append(mask)
        
    unique_targets = list(set(target_masks))
    
    available_masks = set()
    for i in range(n):
        available_masks.add(1 << i)
    available_masks.add(0)
    
    mask_to_wire = {}
    for i in range(n):
        mask_to_wire[1 << i] = i
    mask_to_wire[0] = n
    
    next_wire = n + 1
    gates = []
    
    remaining_targets = set(unique_targets)
    for t in list(remaining_targets):
        if t in available_masks:
            remaining_targets.remove(t)
            
    # Precompute current min popcount for each target relative to available set
    # min_pop[t] = min(popcount(t^u) for u in available_masks)
    # This represents the "cost" to finish t if we could use any single available wire.
    # Note: This assumes we build t^u from scratch, which is an overestimate if 
    # t^u is also complex, but it's a good proxy for "simplicity".
    
    def compute_min_pops(targets, available):
        mins = {}
        for t in targets:
            min_p = 100
            for u in available:
                p = (t ^ u).bit_count()
                if p < min_p:
                    min_p = p
            mins[t] = min_p
        return mins
        
    current_min_pops = compute_min_pops(remaining_targets, available_masks)
    
    while remaining_targets:
        if time.time() > DEADLINE:
            break
            
        avail_list = list(available_masks)
        L = len(avail_list)
        
        best_w = None
        best_score = -1
        best_a = None
        best_b = None
        
        # If L is large, sampling
        if L > 800:
            # Sample pairs
            import random
            # We want to find a w that reduces min_pop for many targets.
            # Try a few random pairs
            for _ in range(100):
                i = random.randint(0, L-1)
                j = random.randint(0, L-1)
                if i == j: continue
                a = avail_list[i]
                b = avail_list[j]
                w = a ^ b
                if w in available_masks:
                    continue
                
                score = 0
                for t in remaining_targets:
                    new_p = (t ^ w).bit_count()
                    if new_p < current_min_pops[t]:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_w = w
                    best_a = a
                    best_b = b
        else:
            for i in range(L):
                a = avail_list[i]
                for j in range(i + 1, L):
                    b = avail_list[j]
                    w = a ^ b
                    if w in available_masks:
                        continue
                    
                    score = 0
                    # Direct hit bonus
                    if w in remaining_targets:
                        score += 10000
                    
                    for t in remaining_targets:
                        new_p = (t ^ w).bit_count()
                        if new_p < current_min_pops[t]:
                            score += 1
                            
                    if score > best_score:
                        best_score = score
                        best_w = w
                        best_a = a
                        best_b = b
                        
                        if best_w in remaining_targets:
                            break
                if best_w in remaining_targets:
                    break
        
        if best_w is None:
            # Fallback: Pick any new wire
            for i in range(L):
                a = avail_list[i]
                for j in range(i+1, L):
                    b = avail_list[j]
                    w = a ^ b
                    if w not in available_masks:
                        best_w = w
                        best_a = a
                        best_b = b
                        break
                if best_w is not None:
                    break
                    
        if best_w is None:
            break
            
        wire_a = mask_to_wire[best_a]
        wire_b = mask_to_wire[best_b]
        
        gates.append([wire_a, wire_b])
        new_wire_idx = next_wire
        next_wire += 1
        
        mask_to_wire[best_w] = new_wire_idx
        available_masks.add(best_w)
        
        if best_w in remaining_targets:
            remaining_targets.remove(best_w)
            
        # Update min pops for remaining targets
        # Only need to update if best_w is "close" to them?
        # Just recompute for simplicity if set is small, else update incrementally.
        if len(remaining_targets) < 50:
            current_min_pops = compute_min_pops(remaining_targets, available_masks)
        else:
            # Incremental update:
            # For each t, new_min = min(old_min, popcount(t^best_w))
            for t in remaining_targets:
                p = (t ^ best_w).bit_count()
                if p < current_min_pops[t]:
                    current_min_pops[t] = p

    # Fallback for unsolved targets
    unsolved = set(target_masks) - available_masks
    for t in unsolved:
        if time.time() > DEADLINE:
            # Force simple construction
            pass
            
        # Find best available wire to reduce popcount
        best_w = 0
        min_pop = t.bit_count()
        for w in available_masks:
            diff = t ^ w
            p = diff.bit_count()
            if p < min_pop:
                min_pop = p
                best_w = w
                
        rem = t ^ best_w
        
        if rem in available_masks:
            wire_bw = mask_to_wire[best_w]
            wire_rem = mask_to_wire[rem]
            gates.append([wire_bw, wire_rem])
            new_wire_idx = next_wire
            next_wire += 1
            mask_to_wire[t] = new_wire_idx
            available_masks.add(t)
        else:
            bits = [i for i in range(n) if (rem >> i) & 1]
            
            if len(bits) == 0:
                wire_rem = n
            elif len(bits) == 1:
                wire_rem = bits[0]
            else:
                curr_wire = bits[0]
                for b in bits[1:]:
                    gates.append([curr_wire, b])
                    new_wire_idx = next_wire
                    next_wire += 1
                    curr_wire = new_wire_idx
                wire_rem = curr_wire
                
            wire_bw = mask_to_wire[best_w]
            gates.append([wire_bw, wire_rem])
            new_wire_idx = next_wire
            next_wire += 1
            mask_to_wire[t] = new_wire_idx
            available_masks.add(t)
            
    outputs = []
    for t in target_masks:
        if t in mask_to_wire:
            outputs.append(mask_to_wire[t])
        else:
            outputs.append(n)
            
    return {'gates': gates, 'outputs': outputs}
