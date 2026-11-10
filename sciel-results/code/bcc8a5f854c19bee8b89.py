"""
Strategy: 
1. Implement a fast DPLL-based SAT solver for 2-colorability of hypergraphs.
   - Variables: v_i (True/False for color 0/1)
   - Constraint: For each edge e, NOT(all 0) AND NOT(all 1).
   - This is equivalent to: For each edge, at least one 0 AND at least one 1.
   - Clauses: For edge {v1...v5}, clause (v1 OR v2 OR v3 OR v4 OR v5) and 
             clause (!v1 OR !v2 OR !v3 OR !v4 OR !v5).
2. Use this fast solver to drive a local search:
   - Start with random sets of edges (n=12 or n=13).
   - Greedily remove edges that are not "critical" (i.e., removing them makes the graph 2-colorable).
   - Repeat with different random seeds to find a minimal non-2-colorable subgraph.
3. Return the smallest non-2-colorable hypergraph found.
"""

import time
import random
from itertools import combinations

def _is_2_colorable_fast(n, edges):
    """
    Checks if a 5-uniform hypergraph on n vertices is 2-colorable.
    Returns True if 2-colorable, False if not.
    Uses DPLL with unit propagation.
    """
    # Convert edges to clauses
    # Clause type 0: positive (at least one 1) -> (v1 OR v2 OR v3 OR v4 OR v5)
    # Clause type 1: negative (at least one 0) -> (!v1 OR !v2 OR !v3 OR !v4 OR !v5)
    
    clauses = []
    for e in edges:
        # Positive clause: all literals positive
        clauses.append((0, tuple(e))) # 0 indicates positive literals
        # Negative clause: all literals negative
        clauses.append((1, tuple(e))) # 1 indicates negative literals
        
    # DPLL state:
    # assignment: dict var -> 0/1, -1 if unassigned
    # We use a simple recursive DPLL with unit propagation.
    
    def propagate(clauses, assignment):
        """
        Performs unit propagation.
        Returns (new_clauses, new_assignment, contradiction_flag)
        """
        new_clauses = []
        new_assignment = dict(assignment)
        changed = True
        contradiction = False
        
        # We'll loop until no more unit clauses
        # For efficiency, we just iterate and filter
        # Since n is small (<=26) and edges <=200, this is fast enough
        
        # To handle unit propagation properly, we need to check if a clause has only one unassigned literal.
        # If so, that literal must be True.
        
        # Simple approach: repeat until stable
        # Given small size, we can just do a few passes or a loop.
        
        while True:
            new_clauses = []
            unit_assigned = False
            for c_type, lits in clauses:
                # lits is a tuple of vertex indices
                # c_type 0: positive literals (v_i)
                # c_type 1: negative literals (!v_i)
                
                unassigned_lits = []
                satisfied = False
                falsified = True # starts as true, if any unassigned or true, becomes false? No.
                
                # Let's evaluate the clause
                # Clause is satisfied if any literal is True.
                # Clause is falsified if all literals are False.
                # Unit clause if exactly one literal is unassigned and others are False.
                
                num_unassigned = 0
                num_false = 0
                unassigned_var = -1
                
                for v in lits:
                    val = new_assignment.get(v, -1)
                    if c_type == 0: # Positive literal v
                        if val == 1:
                            satisfied = True
                            break
                        elif val == 0:
                            num_false += 1
                        else:
                            num_unassigned += 1
                            unassigned_var = v
                    else: # Negative literal !v
                        if val == 0:
                            satisfied = True
                            break
                        elif val == 1:
                            num_false += 1
                        else:
                            num_unassigned += 1
                            unassigned_var = v
                
                if satisfied:
                    continue # Clause is satisfied, drop it
                
                if num_unassigned == 0:
                    # All literals are False
                    contradiction = True
                    break
                
                if num_unassigned == 1:
                    # Unit clause
                    # The unassigned variable must take a value to satisfy the clause
                    if c_type == 0:
                        # Literal is v, so v must be 1
                        new_assignment[unassigned_var] = 1
                    else:
                        # Literal is !v, so v must be 0
                        new_assignment[unassigned_var] = 0
                    unit_assigned = True
                    # This clause is now satisfied (by the assignment), so we drop it
                    # Wait, if we assign it, the clause is satisfied. So we don't add it to new_clauses.
                    continue
                
                # If num_unassigned > 1, keep the clause
                new_clauses.append((c_type, lits))
            
            if contradiction:
                break
            
            if not unit_assigned:
                break # No more unit propagation possible
        
        return new_clauses, new_assignment, contradiction

    def dpll(clauses, assignment):
        # Check for contradiction first (empty clause)
        # propagate handles this
        
        clauses, assignment, contradiction = propagate(clauses, assignment)
        if contradiction:
            return False
        
        if not clauses:
            return True
        
        # Choose a variable
        # Heuristic: pick variable in the shortest clause
        min_len = 100
        var_to_pick = -1
        for c_type, lits in clauses:
            for v in lits:
                if assignment.get(v, -1) == -1:
                    if len(lits) < min_len:
                        min_len = len(lits)
                        var_to_pick = v
                    break # Just pick one from this clause? Or scan all?
            # Better heuristic: find variable with highest occurrence in short clauses
            # For now, simple pick is fine for small n
        
        if var_to_pick == -1:
            # All variables assigned?
            # If clauses remain, they must be satisfied (checked in propagate)
            return True
            
        # Branch
        # Try 0
        assignment[var_to_pick] = 0
        if dpll(clauses, assignment):
            return True
        
        # Try 1
        assignment[var_to_pick] = 1
        if dpll(clauses, assignment):
            return True
        
        # Backtrack
        del assignment[var_to_pick]
        return False

    initial_assignment = {}
    return dpll(clauses, initial_assignment)

def _remove_critical_edges(n, edges):
    """
    Greedily removes edges that are not critical.
    An edge is critical if removing it makes the hypergraph 2-colorable.
    If removing it keeps it non-2-colorable, we remove it.
    """
    current_edges = list(edges)
    # Shuffle to avoid order bias? Or just iterate.
    # We want to find a minimal set.
    # If we remove an edge and the graph becomes 2-colorable, we must keep it.
    # If we remove an edge and it remains non-2-colorable, we can remove it.
    
    # To make it deterministic but effective, we can shuffle.
    random.shuffle(current_edges)
    
    i = 0
    while i < len(current_edges):
        e = current_edges[i]
        test_edges = [x for x in current_edges if x != e]
        
        # Check if test_edges is 2-colorable
        if _is_2_colorable_fast(n, test_edges):
            # It became colorable, so e is critical. Keep it.
            i += 1
        else:
            # It remained non-colorable, so e is redundant. Remove it.
            current_edges.pop(i)
            # Do not increment i, check the next edge at same index
            
    return current_edges

def construct():
    start_time = time.time()
    deadline = start_time + 140 # Safety margin
    
    best_edges = None
    best_len = float('inf')
    
    # Strategy: Try different n and random initializations
    # Known bounds: m(5) <= 51. 
    # n=12 has C(12,5) = 792 edges.
    # n=13 has C(13,5) = 1287 edges.
    # n=10 has C(10,5) = 252 edges.
    
    # We will try to find a small non-2-colorable subgraph.
    # Start with n=12, random subset of edges.
    # If we can't find a good one, try n=11, n=10, etc.
    
    for n in [12, 13, 11, 14]:
        if time.time() > deadline:
            break
            
        all_edges = list(combinations(range(n), 5))
        random.shuffle(all_edges)
        
        # Try multiple random subsets
        for attempt in range(20):
            if time.time() > deadline:
                break
                
            # Start with a random subset of size ~ 60-80
            # If too small, it might be colorable. If too large, we have to remove many.
            # Let's start with a random set of size 80.
            k = 80
            if k > len(all_edges):
                k = len(all_edges)
            
            initial_edges = all_edges[:k]
            
            # First, check if this random set is non-2-colorable
            # If it is colorable, we can't remove edges to make it non-colorable.
            # We need a starting point that is NON-2-colorable.
            
            # How to find a non-2-colorable set?
            # Random sets are usually colorable for small k.
            # We need to add edges until it becomes non-colorable, or start with a known non-colorable structure.
            
            # Alternative: Start with ALL edges (complete hypergraph) which is definitely non-colorable for n>=5?
            # Wait, K^5_5 is non-colorable? 
            # Edges: {0,1,2,3,4}. 
            # Colorings: 32. 
            # All 5 same color: 2 colorings (all 0 or all 1). 
            # If we have only 1 edge, it's colorable (e.g., 0,0,0,0,1).
            # So K^5_5 IS colorable.
            # K^5_6? 
            # Generally, complete hypergraphs K^k_n are colorable if n > k?
            # Actually, Property B is about existence of a 2-coloring.
            # For K^k_n, it is 2-colorable if and only if... 
            # It's known that K^k_n is 2-colorable for n >= 2k? No.
            # Let's just generate random edges and check.
            
            # Better approach:
            # 1. Generate a random set of edges.
            # 2. Check if it's 2-colorable.
            # 3. If yes, add more edges randomly until it becomes non-2-colorable.
            # 4. Once non-2-colorable, run _remove_critical_edges.
            
            current_edges = []
            is_non_colorable = False
            
            # Add edges one by one until non-colorable
            for e in all_edges:
                current_edges.append(e)
                if _is_2_colorable_fast(n, current_edges):
                    continue
                else:
                    is_non_colorable = True
                    break
            
            if not is_non_colorable:
                continue
                
            # Now we have a non-2-colorable set.
            # Minimize it.
            minimized_edges = _remove_critical_edges(n, current_edges)
            
            if len(minimized_edges) < best_len:
                best_len = len(minimized_edges)
                best_edges = minimized_edges
                # If we hit the theoretical lower bound or a very good number, we might stop early?
                # But let's keep searching to see if we can beat 51.
                
            # If we found a very small one, maybe try to polish?
            # For now, just update best.
            
    if best_edges is None:
        # Fallback: Return a known non-colorable hypergraph
        # e.g. Fano plane analog? No, that's 3-uniform.
        # Let's construct a simple non-colorable 5-uniform hypergraph.
        # If n=5, 1 edge is colorable.
        # If n=6, can we make it non-colorable?
        # Let's just return a valid one if we found one, else try n=9 complete (126 edges) which is definitely non-colorable?
        # Wait, is K^5_9 non-colorable?
        # Yes, likely.
        # But we want to minimize.
        # If best_edges is None, it means our search failed.
        # We should return a valid non-colorable hypergraph.
        # Let's use the seed's logic as a fallback: n=9, all edges.
        n = 9
        edges = list(combinations(range(n), 5))
        best_edges = edges
        best_len = len(edges)
        
    return {'n': n, 'edges': [list(e) for e in best_edges]}

# To ensure the fallback n is correct if best_edges comes from a specific n
# We need to track n with best_edges.

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_edges = None
    best_n = None
    best_len = float('inf')
    
    for n in [12, 13, 11, 14, 10]:
        if time.time() > deadline:
            break
            
        all_edges = list(combinations(range(n), 5))
        random.shuffle(all_edges)
        
        for attempt in range(30):
            if time.time() > deadline:
                break
                
            current_edges = []
            is_non_colorable = False
            
            # We need to ensure we don't just pick the first non-colorable one, 
            # as it might have many redundant edges.
            # But _remove_critical_edges will handle that.
            
            for e in all_edges:
                current_edges.append(e)
                if _is_2_colorable_fast(n, current_edges):
                    continue
                else:
                    is_non_colorable = True
                    break
            
            if not is_non_colorable:
                continue
                
            minimized_edges = _remove_critical_edges(n, current_edges)
            
            if len(minimized_edges) < best_len:
                best_len = len(minimized_edges)
                best_edges = minimized_edges
                best_n = n
                
    if best_edges is None:
        n = 9
        edges = list(combinations(range(n), 5))
        best_edges = edges
        best_n = n
        best_len = len(edges)
        
    return {'n': best_n, 'edges': [list(e) for e in best_edges]}

# Re-define construct to avoid shadowing issues if I defined it twice above in my thought process.
# The code block should contain the final version.

import time
import random
from itertools import combinations

def _is_2_colorable_fast(n, edges):
    """
    Checks if a 5-uniform hypergraph on n vertices is 2-colorable.
    Returns True if 2-colorable, False if not.
    """
    clauses = []
    for e in edges:
        clauses.append((0, tuple(e)))
        clauses.append((1, tuple(e)))
        
    def propagate(clauses, assignment):
        new_clauses = []
        new_assignment = dict(assignment)
        contradiction = False
        changed = True
        
        while changed:
            changed = False
            new_clauses = []
            for c_type, lits in clauses:
                unassigned_lits = []
                satisfied = False
                num_unassigned = 0
                num_false = 0
                unassigned_var = -1
                
                for v in lits:
                    val = new_assignment.get(v, -1)
                    if c_type == 0:
                        if val == 1:
                            satisfied = True
                            break
                        elif val == 0:
                            num_false += 1
                        else:
                            num_unassigned += 1
                            unassigned_var = v
                    else:
                        if val == 0:
                            satisfied = True
                            break
                        elif val == 1:
                            num_false += 1
                        else:
                            num_unassigned += 1
                            unassigned_var = v
                
                if satisfied:
                    continue
                
                if num_unassigned == 0:
                    contradiction = True
                    break
                
                if num_unassigned == 1:
                    if c_type == 0:
                        new_assignment[unassigned_var] = 1
                    else:
                        new_assignment[unassigned_var] = 0
                    changed = True
                    continue
                
                new_clauses.append((c_type, lits))
            
            if contradiction:
                break
        
        return new_clauses, new_assignment, contradiction

    def dpll(clauses, assignment):
        clauses, assignment, contradiction = propagate(clauses, assignment)
        if contradiction:
            return False
        
        if not clauses:
            return True
        
        # Pick variable
        var_to_pick = -1
        min_len = 100
        for c_type, lits in clauses:
            for v in lits:
                if assignment.get(v, -1) == -1:
                    if len(lits) < min_len:
                        min_len = len(lits)
                        var_to_pick = v
                    break
        
        if var_to_pick == -1:
            return True
            
        assignment[var_to_pick] = 0
        if dpll(clauses, assignment):
            return True
        
        assignment[var_to_pick] = 1
        if dpll(clauses, assignment):
            return True
        
        del assignment[var_to_pick]
        return False

    return dpll(clauses, {})

def _remove_critical_edges(n, edges):
    current_edges = list(edges)
    random.shuffle(current_edges)
    
    i = 0
    while i < len(current_edges):
        e = current_edges[i]
        test_edges = [x for x in current_edges if x != e]
        
        if _is_2_colorable_fast(n, test_edges):
            i += 1
        else:
            current_edges.pop(i)
            
    return current_edges

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_edges = None
    best_n = None
    best_len = float('inf')
    
    for n in [12, 13, 11, 14, 10]:
        if time.time() > deadline:
            break
            
        all_edges = list(combinations(range(n), 5))
        random.shuffle(all_edges)
        
        for attempt in range(30):
            if time.time() > deadline:
                break
                
            current_edges = []
            is_non_colorable = False
            
            for e in all_edges:
                current_edges.append(e)
                if _is_2_colorable_fast(n, current_edges):
                    continue
                else:
                    is_non_colorable = True
                    break
            
            if not is_non_colorable:
                continue
                
            minimized_edges = _remove_critical_edges(n, current_edges)
            
            if len(minimized_edges) < best_len:
                best_len = len(minimized_edges)
                best_edges = minimized_edges
                best_n = n
                
    if best_edges is None:
        n = 9
        edges = list(combinations(range(n), 5))
        best_edges = edges
        best_n = n
        best_len = len(edges)
        
    return {'n': best_n, 'edges': [list(e) for e in best_edges]}

