"""Rewritten program: local search on random graphs with efficient violation counting."""
import random
import time
import itertools
import math

def _count_violations(adj, n):
    """
    Count the number of induced subgraphs of size >= 6 that are regular.
    Returns the total count (0 means valid).
    Uses bitmask adjacency for fast degree computation.
    """
    total = 0
    # Check k from 6 to n
    for k in range(6, n + 1):
        for subset in itertools.combinations(range(n), k):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            
            # Compute degrees of all vertices in subset
            is_reg = True
            first_deg = -1
            for v in subset:
                d = (adj[v] & mask).bit_count()
                if first_deg == -1:
                    first_deg = d
                elif d != first_deg:
                    is_reg = False
                    break
            
            if is_reg:
                total += 1
                # Early exit if we just need to know if any exist? 
                # No, for local search we want a score to minimize.
                # But if total > 0, we can stop if we only care about validity?
                # Actually, for local search, minimizing the count is better.
                # However, counting ALL is expensive.
                # Let's just return 1 if we find any? No, that's a step function.
                # A smoother landscape is better.
                # But for n=20, checking all subsets is C(20,6)+... which is huge.
                # C(20,6) = 38760. C(20,7)=77520. C(20,8)=125970...
                # Total subsets of size >=6 is ~ 2^20 - (1+20+190+1140+4845+15504) = 1048576 - 21700 ~ 1M.
                # Iterating 1M subsets and doing bit ops is fast in Python? 
                # 1M iterations * ~10 ops = 10M ops. Should be < 1s.
                # But we do this in a local search loop. If we do 1000 steps, that's 10^9 ops -> too slow.
                
                # Optimization: We can stop counting once total exceeds a threshold?
                # Or we can just check if ANY exist for validity, and use a different heuristic for search?
                # Actually, the inspirations suggest "local_search".
                # A common approach: Minimize the number of regular induced subgraphs.
                # But if it's too slow, we can just check for existence (0/1).
                # If we use 0/1, the landscape is rugged.
                # Let's try to count, but with an early break if count gets high?
                # Or better: Just check existence. If we find one, we know it's invalid.
                # For local search, we want to move towards valid.
                # If we flip an edge and the count goes from 5 to 4, that's good.
                # But if we just check existence, we go from 1 to 1 (if still invalid) or 1 to 0 (if valid).
                # This is a hill-climbing on a step function. It might work but get stuck.
                
                # Let's stick to counting for now, but optimize the inner loop.
                # Actually, for n=20, C(20,6) is the dominant term for small k.
                # Let's just count.
                
    return total

def _is_valid(adj, n):
    """Check if graph is valid (no regular induced subgraph of size >= 6)."""
    for k in range(6, n + 1):
        for subset in itertools.combinations(range(n), k):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            
            is_reg = True
            first_deg = -1
            for v in subset:
                d = (adj[v] & mask).bit_count()
                if first_deg == -1:
                    first_deg = d
                elif d != first_deg:
                    is_reg = False
                    break
            
            if is_reg:
                return False
    return True

def _random_graph(n, p):
    """Generate a random graph with edge probability p."""
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _local_search(adj, n, time_limit):
    """
    Perform local search to remove regular induced subgraphs.
    Strategy: 
    1. Start with a random graph.
    2. Find a violation (regular induced subgraph).
    3. Try flipping edges within or near the violation to break it.
    4. Accept if the number of violations decreases (or stays same but we make progress?).
    Since counting all violations is expensive, we will:
    - Find ONE violation.
    - Try to fix it by flipping random edges in the induced subgraph or connecting to outside.
    - Check if the graph is valid. If yes, return.
    - If not, check if the specific violation is gone? 
    Actually, a simpler local search:
    - Start with random graph.
    - While not valid:
        - Find a violating subset S.
        - Pick a random edge (u,v) where u in S, v in S (or maybe u in S, v outside?).
        - Flip it.
        - If the new graph has fewer violations (or is valid), keep it.
        - Else revert.
    To avoid O(N^2) violation counting at every step, we can just check validity.
    But checking validity is the bottleneck.
    
    Alternative: 
    Use a "repair" strategy.
    1. Generate random graph.
    2. Check validity.
    3. If invalid, find a violation S.
    4. Randomly flip an edge in S.
    5. Check if S is still regular. If not, we made progress? Not necessarily, other violations might remain.
    6. Repeat.
    
    Let's implement a simple hill-climbing on the number of violations, but limit the count to speed up.
    If we find more than, say, 10 violations, we stop counting and return a high number.
    This makes the score landscape smoother for small numbers of violations.
    """
    start_time = time.time()
    
    # Initial check
    if _is_valid(adj, n):
        return adj, True
    
    # We will try to minimize violations.
    # To speed up, we'll use a function that returns min(count, CAP)
    CAP = 50
    
    def count_violations_capped(adj, n):
        total = 0
        for k in range(6, n + 1):
            for subset in itertools.combinations(range(n), k):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                
                is_reg = True
                first_deg = -1
                for v in subset:
                    d = (adj[v] & mask).bit_count()
                    if first_deg == -1:
                        first_deg = d
                    elif d != first_deg:
                        is_reg = False
                        break
                
                if is_reg:
                    total += 1
                    if total >= CAP:
                        return total
        return total
    
    current_score = count_violations_capped(adj, n)
    if current_score == 0:
        return adj, True
        
    best_adj = adj[:]
    best_score = current_score
    
    # Local search steps
    # We will try flipping random edges.
    # Number of steps depends on time.
    # Each step involves a check. If check is ~0.1s, we can do ~100 steps in 10s.
    # But for n=20, check is ~0.05s? Let's assume 0.1s.
    # Total time limit for search is ~100s.
    # So ~1000 steps?
    
    # Actually, we can be smarter.
    # Find a violation, flip an edge in it, check if that specific violation is gone.
    # This is much faster than checking all violations.
    # But we might introduce new violations.
    # Let's try:
    # 1. Find violation S.
    # 2. Flip edge (u,v) in S.
    # 3. Check if S is still regular.
    # 4. If S is not regular, we "fixed" this one. But did we create others?
    # 5. Check full validity. If valid, return.
    # 6. If not valid, we have to decide if we keep the flip.
    #    We can keep it if it doesn't increase the "local" violation count?
    #    Or just keep it randomly?
    
    # Let's try a different approach: 
    # Start with a graph that is "almost" valid.
    # Random graphs with p=0.5 have many regular subgraphs.
    # What p is good?
    # For n=20, C(20,6) is large.
    # Probability that a specific 6-set is regular?
    # Degrees in a 6-set induced by G(n, p).
    # If p=0.5, degrees are Bin(5, 0.5).
    # P(all 6 degrees equal) = sum_d P(deg=d)^6.
    # This is small but non-zero.
    # With ~40k subsets, we expect many violations.
    # We need to remove them.
    
    # Let's use the "find violation, fix it" loop.
    
    while time.time() - start_time < time_limit:
        # Find a violation
        viol_subset = None
        for k in range(6, n + 1):
            for subset in itertools.combinations(range(n), k):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                
                is_reg = True
                first_deg = -1
                for v in subset:
                    d = (adj[v] & mask).bit_count()
                    if first_deg == -1:
                        first_deg = d
                    elif d != first_deg:
                        is_reg = False
                        break
                
                if is_reg:
                    viol_subset = subset
                    break
            if viol_subset:
                break
        
        if viol_subset is None:
            return adj, True # Valid!
        
        # Pick a random edge in the violating subset to flip
        # Or pick a random edge in the whole graph?
        # Flipping an edge inside the subset is more likely to break regularity.
        u, v = random.sample(list(viol_subset), 2)
        
        # Flip edge (u, v)
        adj[u] ^= (1 << v)
        adj[v] ^= (1 << u)
        
        # Check if the graph is now valid
        if _is_valid(adj, n):
            return adj, True
        
        # Check if this specific violation is gone
        mask = 0
        for x in viol_subset:
            mask |= (1 << x)
        
        is_reg = True
        first_deg = -1
        for x in viol_subset:
            d = (adj[x] & mask).bit_count()
            if first_deg == -1:
                first_deg = d
            elif d != first_deg:
                is_reg = False
                break
        
        if not is_reg:
            # We fixed this violation. Keep the flip.
            # But we might have created new ones. We'll find them in the next iteration.
            continue
        else:
            # The violation is still there. Revert the flip.
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            # Try a different edge in the subset?
            # Or just break and restart?
            # Let's try a few random edges in the subset before giving up on this subset.
            # Actually, if we flip one edge and it's still regular, it's likely because the degrees adjusted in a way that preserved regularity?
            # E.g. if all degrees were 2, and we add an edge, two degrees become 3, others 2. Not regular.
            # So flipping an edge in a regular induced subgraph *should* break regularity unless the subgraph is very small or specific?
            # Wait. If S is regular with degree d.
            # If we add edge (u,v) in S:
            # deg(u) becomes d+1, deg(v) becomes d+1, others d.
            # Not regular (unless d+1 == d, impossible).
            # If we remove edge (u,v) in S:
            # deg(u) becomes d-1, deg(v) becomes d-1, others d.
            # Not regular.
            # So flipping ANY edge in S should break the regularity of S!
            # Why did my check say it's still regular?
            # Ah, I see. `is_reg` check above uses the *new* adj.
            # If I flip (u,v), the degrees change.
            # So `is_reg` should be False.
            # Unless... I made a mistake in the flip logic?
            # `adj[u] ^= (1 << v)` flips the bit.
            # If the edge existed, it's removed. If not, it's added.
            # In either case, degrees of u and v change by +/- 1.
            # So the degree sequence changes.
            # So S cannot remain regular.
            # So the `if not is_reg` branch should always be taken?
            # Yes. So we always keep the flip if we flip an edge inside S.
            # Then why would we ever revert?
            # We don't need to revert if we flip inside S.
            # So the logic simplifies:
            # 1. Find violation S.
            # 2. Flip a random edge in S.
            # 3. Check full validity.
            # 4. If valid, return.
            # 5. Else, loop.
            
            # Wait, what if flipping an edge in S creates a NEW violation?
            # Yes, it might. But we'll find it in the next iteration.
            # So we don't need to check if S is still regular, because it *can't* be.
            # So we can just flip and continue.
            
            # However, if we flip an edge and it doesn't help (i.e., we still have violations), we might be stuck in a loop?
            # No, because we are changing the graph.
            # But we might oscillate.
            # To avoid oscillation, we can use a history or just hope it converges.
            # Or we can use a "score" based on the number of violations.
            # But counting is slow.
            
            # Let's stick to the simple "flip edge in violation" method.
            # It should converge because each flip breaks at least one violation.
            # The number of violations might increase, but it's unlikely to increase by more than 1?
            # Actually, adding an edge can create many new regular subgraphs.
            # So the number of violations might increase.
            # This is a random walk.
            # To make it a hill-climbing, we should only accept flips that reduce the number of violations.
            # But that requires counting.
            
            # Compromise: 
            # Use the simple flip method. If it doesn't find a valid graph in time, try a different strategy.
            
            # Actually, let's just continue the loop.
            pass

    return adj, False

def construct():
    start_time = time.time()
    deadline = start_time + 140 # Safety margin
    
    best_n = 5
    best_edges = []
    
    target_ns = [21, 20, 19, 18]
    
    for n_target in target_ns:
        if time.time() > deadline - 20:
            break
            
        # Try different p values
        # For n=20, p around 0.5 is standard.
        # But maybe p closer to 0 or 1 is better?
        # Regular subgraphs are more likely in dense or sparse graphs?
        # In a random graph, the expected number of regular induced subgraphs depends on p.
        # Let's try p = 0.3, 0.5, 0.7.
        
        p_values = [0.3, 0.4, 0.5, 0.6, 0.7]
        
        for p in p_values:
            if time.time() > deadline - 10:
                break
                
            # Generate random graph
            adj = _random_graph(n_target, p)
            
            # Check if valid immediately
            if _is_valid(adj, n_target):
                edges = []
                for i in range(n_target):
                    for j in range(i + 1, n_target):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                
                if n_target > best_n:
                    best_n = n_target
                    best_edges = edges
                    
                if n_target == 21:
                    return {'n': best_n, 'edges': best_edges}
                
                # If we found 20, we can break p loop and n loop?
                # Yes, 20 is the best we can hope for if 21 fails.
                break
            
            # If not valid, try local search
            search_time = 10 # seconds
            adj, valid = _local_search(adj, n_target, search_time)
            
            if valid:
                edges = []
                for i in range(n_target):
                    for j in range(i + 1, n_target):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                
                if n_target > best_n:
                    best_n = n_target
                    best_edges = edges
                
                if n_target == 21:
                    return {'n': best_n, 'edges': best_edges}
                
                break # Break p loop
        
        # If we found a valid graph for this n_target, break outer loop
        if best_n == n_target:
            break
            
    return {'n': best_n, 'edges': best_edges}
