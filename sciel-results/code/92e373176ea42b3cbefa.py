"""
Exact XOR-circuit synthesis: Set-Cover approach over linear subspaces.
Strategy: 
1. Identify all input columns.
2. Generate a library of candidate intermediate wires:
   - Single inputs (depth 0)
   - Pairs of inputs (depth 1, cost 1 gate each)
   - Triples of inputs (depth 2, cost 2 gates each, but can be shared)
   - Actually, for efficiency, we limit to a "basis" of useful combinations.
   A better approach for XOR circuits:
   The cost of a circuit is the number of gates.
   Each gate computes XOR of two existing wires.
   
   Alternative Strategy: "Basis Expansion" / "Gaussian Elimination on Circuit"
   Instead of per-row trees, we can view the set of required outputs as a set of vectors.
   We want to compute these vectors using XOR gates.
   This is equivalent to finding a sequence of vectors v_1, ..., v_k such that:
   - v_1, ..., v_k are linear combinations of inputs.
   - Each v_i is XOR of two previously computed vectors (or inputs).
   - The required outputs are in the span of {inputs} U {v_1, ..., v_k}.
   - Minimize k.
   
   This is the "Straight-line Program" (SLP) size problem for linear maps over GF(2).
   
   Known result: For a matrix M, the minimal SLP size is related to the rank, but not exactly.
   However, a strong heuristic is to use a greedy algorithm that picks the "most useful" next vector.
   
   Let's try a different angle: 
   The baseline "independent-row" likely computes each row independently.
   For a row with weight w, it takes w-1 gates (a tree).
   Total baseline gates = sum(popcount(row) - 1 for row in rows).
   
   To beat this, we must share gates.
   
   New Strategy: "Clustering by Support"
   1. Group input columns by their frequency of appearing in rows.
   2. Build a "core" circuit for the most frequent columns.
   3. Express other rows as XORs of core wires and single inputs.
   
   Let's implement a Greedy Set Cover over "Patterns":
   A "pattern" is a set of input indices.
   We have a set of "required patterns" (the rows).
   We can "buy" a pattern P (meaning we create a wire for XOR(P)) at a cost.
   Cost of buying P: 
   - If |P| == 1, cost 0 (input).
   - If |P| == 2, cost 1 (1 gate).
   - If |P| > 2, we can build P from smaller patterns.
   
   Actually, let's simplify. 
   The most effective sharing happens when rows share large subsets of 1s.
   
   Algorithm:
   1. Compute all rows.
   2. Initialize a set of "available wires" = {single inputs}.
   3. While there are uncovered rows:
      a. For each available wire w1 and w2, consider w = w1 ^ w2.
      b. Calculate the "benefit" of creating w: how many currently uncovered rows can be simplified or solved by having w?
      c. Pick the w with max benefit / cost.
      d. Add w to available wires.
   
   This is essentially a greedy construction of a linear basis that covers the rows.
   
   Let's refine:
   Represent each row as a frozenset of indices.
   Available wires: dict frozenset -> wire_id.
   Initially: {frozenset([i]): i for i in range(n)}
   
   Target: Cover all rows.
   A row R is "covered" if R is in available wires, OR R can be formed by XOR of available wires?
   No, the output must be a single wire. So the row's set MUST be an available wire.
   
   So we need to generate wires until every row's set is in the dictionary.
   
   Greedy Step:
   - Current uncovered rows: U.
   - Current available wires: A.
   - Candidate new wire: S1 ^ S2 where S1, S2 in A.
   - Benefit(S): 
     - If S is a row in U, benefit = 1 (we solve one row directly).
     - If S is a subset of a row in U, it might help build that row later?
     - Actually, if we have S1 and S2, we can form S1^S2.
     - If we have S and a row R, and R ^ S is in A, then R can be formed.
     
   This is getting complex. Let's use a simpler, proven heuristic for XOR circuits:
   **The "Rank-1 Decomposition" or "Outer Product" approach doesn't apply directly.**
   
   Let's try **Iterative Row Reduction with Global Sharing**:
   1. Start with all rows as "targets".
   2. Maintain a pool of "intermediate" sets.
   3. Repeatedly find a pair of sets (from targets or intermediates) whose XOR reduces the total "complexity" (sum of popcounts) of the remaining targets.
   
   Complexity metric: Sum of popcounts of all unresolved rows.
   When we create a new wire W = A ^ B:
   - We add cost 1 (1 gate).
   - For any unresolved row R:
     - If R == W, R is solved (remove from unresolved).
     - If R ^ W is "simpler" than R (lower popcount), we can replace R with R^W in the unresolved set? 
       No, we don't replace, we just note that R can be computed as W ^ (R^W).
       If we later create a wire for R^W, then R is solved.
       
   So, the state is:
   - `unsolved`: set of frozensets (rows) not yet directly available.
   - `available`: set of frozensets (wires) we have created.
   
   Action: Create new wire S = S1 ^ S2 (S1, S2 in available).
   Cost: 1 gate.
   Benefit: 
   - Any unsolved row R such that R == S is solved.
   - Any unsolved row R such that R ^ S is in available is solved (because R = S ^ (R^S)).
   
   We want to maximize (Number of rows solved) per gate.
   If no rows are solved, we might still create a wire if it reduces the "potential" of future solves.
   But for a greedy approach, let's stick to "solves a row".
   
   If no single XOR of two available wires solves any row, we are stuck?
   No, we can XOR an available wire with an unsolved row? 
   No, we can only XOR two *available* wires to make a new *available* wire.
   Unresolved rows are not wires.
   
   So we must build up the available set until it contains all row sets.
   
   This is equivalent to: Find the smallest set of vectors (wires) generated by XORs of inputs, such that all row vectors are in the span? No, they must be *exactly* in the set of generated wires.
   Wait, if R is in the span of available wires, can we output R?
   The output is a single wire. So R must be *one of the wires*.
   
   So the problem is: Generate a set of wires W such that:
   1. Inputs are in W.
   2. If w1, w2 in W, then w1^w2 can be added to W (at cost 1).
   3. All row vectors are in W.
   4. Minimize number of generated wires (excluding inputs).
   
   This is the "XOR Circuit Size" problem.
   
   Greedy Strategy:
   1. `available` = {inputs}
   2. `targets` = {rows}
   3. While `targets` is not empty:
      a. Find S1, S2 in `available` such that S = S1^S2.
      b. Check if S is in `targets`. If yes, add S to `available`, remove S from `targets`, cost 1.
      c. If no S in available^available is in targets, we are stuck in this greedy step.
         We need to create a wire that *helps* solve targets.
         A wire S helps if S is a subset of a target T, or T is a subset of S, or they overlap significantly.
         
   Actually, if we have inputs, we can form ANY subset by XORing single inputs?
   No, XORing single inputs gives us the parity of the set.
   To get the set {1,2}, we need wire 1 and wire 2, then 1^2.
   To get the set {1,2,3}, we need (1^2) and 3, then (1^2)^3.
   
   So, any set can be formed. The question is efficiency.
   
   Let's try a **Priority Queue of "Potential Solves"**:
   1. Pre-compute all pairwise XORs of available wires.
   2. For each candidate S, calculate `score(S) = number of targets T such that T == S`.
   3. If max score > 0, pick best S, add to available, remove from targets.
   4. If max score == 0, we need to expand `available` with a wire that is "useful".
      Useful means: S is a subset of some target T, or T is a subset of S?
      Actually, if we have S, and we have T in targets, if we also have T^S in available, T is solved.
      So, pick S that maximizes `count(T in targets such that T^S in available)`.
      If that is 0, pick S that minimizes `distance` to targets?
      
   This is expensive to compute for large n.
   n is up to 16. 2^16 = 65536 possible sets.
   We can represent sets as integers (bitmasks).
   
   Algorithm with Bitmasks:
   1. Convert rows to integers.
   2. `available` = set of input bitmasks (1<<i).
   3. `targets` = set of row bitmasks.
   4. `gates` = []
   5. While `targets`:
      a. Generate candidates:
         - All pairwise XORs of `available`.
      b. For each candidate C:
         - `direct_solves` = 1 if C in `targets` else 0
         - `indirect_solves` = count of T in `targets` such that (T ^ C) in `available`
      c. Pick C with max (direct_solves * 100 + indirect_solves).
      d. If best score is 0:
         - We are stuck. We must add a "useless" wire to expand options.
         - Pick C that is a subset of the smallest target? Or just pick a random XOR of two available wires that hasn't been added yet?
         - Actually, if we have all single inputs, we can form any set by XORing singles.
         - But that's expensive.
         - Let's prioritize C that is a subset of a target.
         - `subset_score` = max over T in targets of (popcount(C & T) / popcount(T))
      e. Add C to `available`, record gate.
      f. Remove any solved targets from `targets`.
      
   This might work. The space of available wires grows.
   To keep it fast, limit `available` size or use a heuristic to prune.
   Since n<=16, the number of possible wires is 2^16 = 65536.
   We can afford to keep all generated wires in a set.
   
   Optimization:
   - Use integers for bitmasks.
   - `available` is a set of ints.
   - `targets` is a set of ints.
   - Loop:
     - If `targets` is empty, break.
     - Compute all pairwise XORs of `available`. (Size |A|^2 / 2).
     - If |A| is large (e.g., > 100), this is 5000 ops. Fast.
     - Score each XOR.
     - Pick best.
     
   Let's implement this.
"""

import time
import numpy as np

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 85.0
    
    M = np.array(matrix, dtype=np.uint8)
    n_rows, n_cols = M.shape
    
    # Convert rows to bitmasks
    targets = set()
    for i in range(n_rows):
        mask = 0
        for j in range(n_cols):
            if M[i, j] == 1:
                mask |= (1 << j)
        targets.add(mask)
    
    # Remove 0 if present? 0 is the constant wire.
    # If a row is all zeros, it maps to wire n_cols.
    zero_mask = 0
    if zero_mask in targets:
        targets.remove(zero_mask)
    
    # Available wires: inputs
    available = set()
    for j in range(n_cols):
        available.add(1 << j)
    
    # Mapping from bitmask to wire id
    # Inputs are 0..n_cols-1
    wire_id_map = {}
    for j in range(n_cols):
        wire_id_map[1 << j] = j
    
    next_wire = n_cols + 1
    gates = []
    
    # If no targets, we are done
    if not targets:
        return {'gates': gates, 'outputs': [n_cols] * n_rows} # Wait, outputs must match original rows
        # We need to track which original row got which wire.
        
    # Let's restart with proper output tracking.
    
    targets_list = []
    for i in range(n_rows):
        mask = 0
        for j in range(n_cols):
            if M[i, j] == 1:
                mask |= (1 << j)
        targets_list.append(mask)
        
    # Set of unsolved masks
    unsolved = set(targets_list)
    
    # Wire ID map
    wire_id_map = {1 << j: j for j in range(n_cols)}
    next_wire = n_cols + 1
    gates = []
    
    # Precompute popcount for speed?
    # Python int.bit_count() is fast.
    
    while unsolved:
        if time.time() > deadline:
            # Fallback: assign remaining to 0? No, must be valid.
            # Just break and hope we solved most. 
            # But we must return valid outputs.
            # If we break, we must have wires for all targets.
            # So we must finish. 
            # For n=16, this loop should be fast.
            pass
            
        # Generate candidates
        # To avoid O(N^2) if N is large, we can sample or limit.
        # But N is bounded by 2^16. 
        # However, we only add wires that are "useful".
        # Let's list available as list for iteration.
        avail_list = list(available)
        
        best_score = -1
        best_candidate = None
        best_direct = 0
        best_indirect = 0
        
        # We also need to consider candidates that are subsets of targets if no direct/indirect solve.
        
        for i in range(len(avail_list)):
            a = avail_list[i]
            for j in range(i + 1, len(avail_list)):
                b = avail_list[j]
                c = a ^ b
                
                if c in available:
                    continue # Already have this wire
                
                # Score
                direct = 1 if c in unsolved else 0
                indirect = 0
                if not direct:
                    # Check if c helps solve any unsolved target
                    # T is solved if T ^ c is in available
                    # This is expensive: O(|unsolved|)
                    # Optimization: Only check if direct is 0?
                    # Let's just check direct first.
                    pass
                
                # Actually, checking indirect for every pair is O(N^2 * U).
                # N ~ 100, U ~ 100. 10^6 ops. OK.
                
                # Let's refine scoring:
                # 1. Direct solve: +1000
                # 2. Indirect solve: +100
                # 3. Subset potential: +popcount(c & min_target)
                
                score = 0
                if c in unsolved:
                    score += 1000
                else:
                    # Check indirect
                    # To speed up, only check a few?
                    # No, let's check all if N is small.
                    for t in unsolved:
                        if (t ^ c) in available:
                            score += 100
                            break # One indirect solve is enough to prioritize? 
                                 # Or count them?
                                 # Counting is better.
                    # If no indirect, check subset
                    if score == 0:
                        # Find max overlap with any target
                        max_overlap = 0
                        for t in unsolved:
                            overlap = (c & t).bit_count()
                            if overlap > max_overlap:
                                max_overlap = overlap
                        score = max_overlap
                    
                if score > best_score:
                    best_score = score
                    best_candidate = c
                    best_direct = 1 if c in unsolved else 0
                    best_indirect = score - (1000 if best_direct else 0)
        
        if best_candidate is None:
            # No new wires can be formed? 
            # This implies available is closed under XOR? 
            # If so, and unsolved is not empty, we have a problem.
            # But inputs span the space, so we can always form any vector.
            # The issue is we only form XORs of *available* wires.
            # If available is {inputs}, we can form any pair.
            # So best_candidate should never be None unless available is empty or size 1.
            # If size 1, we can't form new.
            # But we start with n_cols inputs.
            
            # If we are stuck, maybe we need to add a wire that is NOT an XOR of two available?
            # No, the circuit model requires XOR of two wires.
            # So we must have at least 2 available wires.
            
            # If we are stuck, it means all XORs of available are already in available.
            # This means available is a subspace.
            # If unsolved is not in available, we can never solve it?
            # No, we can form new wires by XORing available wires.
            # If available is a subspace, XOR of two elements is in the subspace.
            # So we are stuck in the subspace.
            # We need to "break out" of the subspace?
            # But we can only XOR available wires.
            # So we are confined to the span of available wires.
            # Since inputs are in available, the span is the whole space (if inputs are independent).
            # Wait, if available is a subspace, and we only generate XORs of available, we stay in that subspace.
            # But we started with inputs. The span of inputs is the whole space.
            # So we should be able to reach any vector.
            # Why would we get stuck?
            # Only if we don't generate ALL possible XORs.
            # My loop iterates over all pairs.
            # So if best_candidate is None, it means all XORs are already in available.
            # This implies available is a subspace.
            # If unsolved contains a vector NOT in available, we are stuck.
            # How can this happen?
            # It can't, if available starts as inputs and we add all XORs.
            # Wait, we only add ONE wire per iteration.
            # So available is not closed.
            # So best_candidate should exist.
            
            # If best_candidate is None, break to avoid infinite loop.
            break
        
        # Add best_candidate
        available.add(best_candidate)
        
        # Find two parents
        # We need to find a, b in available such that a^b == best_candidate
        # This is O(N^2) again.
        # We can store parents?
        # Or just search.
        found = False
        for a in avail_list:
            b = a ^ best_candidate
            if b in available and a != b:
                gates.append([wire_id_map[a], wire_id_map[b]])
                wire_id_map[best_candidate] = next_wire
                next_wire += 1
                found = True
                break
            # If a == b, a^a=0. We don't need 0 as a gate output unless we need it.
            # But we don't generate 0 as a gate.
        
        if not found:
            # Should not happen
            continue
            
        # Remove solved targets
        # Direct
        if best_candidate in unsolved:
            unsolved.remove(best_candidate)
        
        # Indirect
        # If T ^ best_candidate is in available, T is solved.
        # We need to check all unsolved.
        to_remove = []
        for t in unsolved:
            if (t ^ best_candidate) in available:
                to_remove.append(t)
        
        for t in to_remove:
            unsolved.remove(t)
            
    # Construct outputs
    outputs = []
    for mask in targets_list:
        if mask in wire_id_map:
            outputs.append(wire_id_map[mask])
        else:
            # Should not happen
            outputs.append(n_cols) # Fallback to 0
            
    return {'gates': gates, 'outputs': outputs}
