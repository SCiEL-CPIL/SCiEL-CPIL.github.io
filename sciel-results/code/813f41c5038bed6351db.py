"""Construct a graph with no regular induced subgraph on 6+ vertices."""
import random
import itertools
import time

def _check_regular_subset(adj, n, subset):
    """Check if the induced subgraph on 'subset' is regular."""
    k = len(subset)
    if k < 6:
        return False
    # Compute degrees within the subset
    idx_map = {v: i for i, v in enumerate(subset)}
    degrees = [0] * k
    for i in range(k):
        u = subset[i]
        for j in range(k):
            if i != j and adj[u][subset[j]]:
                degrees[i] += 1
    # Check if all degrees are equal
    if all(d == degrees[0] for d in degrees):
        return True
    return False

def _is_valid(adj, n):
    """Check if no subset of size >= 6 induces a regular subgraph."""
    for k in range(6, n + 1):
        for subset in itertools.combinations(range(n), k):
            if _check_regular_subset(adj, n, subset):
                return False
    return True

def _is_valid_fast(adj, n):
    """Faster check: only check size 6 first (most likely to fail), then larger."""
    # Check all 6-subsets
    for subset in itertools.combinations(range(n), 6):
        if _check_regular_subset(adj, n, subset):
            return False
    # Check all 7-subsets
    for subset in itertools.combinations(range(n), 7):
        if _check_regular_subset(adj, n, subset):
            return False
    # Check all 8-subsets
    for subset in itertools.combinations(range(n), 8):
        if _check_regular_subset(adj, n, subset):
            return False
    # Check all 9-subsets
    for subset in itertools.combinations(range(n), 9):
        if _check_regular_subset(adj, n, subset):
            return False
    # Check all 10-subsets
    for subset in itertools.combinations(range(n), 10):
        if _check_regular_subset(adj, n, subset):
            return False
    # For larger subsets, check if they contain a regular 6-subset
    # Actually we already checked all 6-subsets, so if a larger subset is regular,
    # it doesn't necessarily contain a regular 6-subset. We need to check them.
    for k in range(11, n + 1):
        for subset in itertools.combinations(range(n), k):
            if _check_regular_subset(adj, n, subset):
                return False
    return True

def _count_violations(adj, n):
    """Count number of regular subsets of size >= 6."""
    count = 0
    for k in range(6, n + 1):
        for subset in itertools.combinations(range(n), k):
            if _check_regular_subset(adj, n, subset):
                count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    best_n = 5
    best_edges = []
    
    # Try to find valid graphs for n = 20 down to 6
    for target_n in range(20, 5, -1):
        if time.time() > deadline - 30:
            break
        
        # Try random graphs with various edge probabilities
        found = False
        attempts = 0
        max_attempts = 50000
        
        while attempts < max_attempts and time.time() < deadline - 30:
            attempts += 1
            
            # Try different p values
            p = 0.5
            
            # Build random graph
            adj = [[False] * target_n for _ in range(target_n)]
            edges = []
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if random.random() < p:
                        adj[i][j] = True
                        adj[j][i] = True
                        edges.append([i, j])
            
            # Quick check: count violations for size 6 only first
            violations_6 = 0
            for subset in itertools.combinations(range(target_n), 6):
                if _check_regular_subset(adj, target_n, subset):
                    violations_6 += 1
                    if violations_6 > 50:
                        break
            
            if violations_6 == 0:
                # Full check
                if _is_valid(adj, target_n):
                    best_n = target_n
                    best_edges = edges
                    found = True
                    break
            
            if attempts % 1000 == 0:
                # Try perturbation: flip some edges
                pass
        
        if found:
            break
    
    # If we didn't find a large one, try a constructive approach
    if best_n < 20:
        # Try a specific construction: use a "blow-up" of a small graph
        # Or try random with local repair
        
        for target_n in range(max(best_n + 1, 15), 21):
            if time.time() > deadline - 20:
                break
            
            # Start with random graph and repair
            adj = [[False] * target_n for _ in range(target_n)]
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if random.random() < 0.5:
                        adj[i][j] = True
                        adj[j][i] = True
            
            # Repair: iteratively fix violations
            for iteration in range(200):
                if time.time() > deadline - 15:
                    break
                
                # Find a violation
                violation_found = False
                for k in range(6, target_n + 1):
                    for subset in itertools.combinations(range(target_n), k):
                        if _check_regular_subset(adj, target_n, subset):
                            # Try to break this regular subgraph
                            # Flip a random edge within the subset
                            i, j = random.sample(subset, 2)
                            adj[i][j] = not adj[i][j]
                            adj[j][i] = adj[i][j]
                            violation_found = True
                            break
                    if violation_found:
                        break
                
                if not violation_found:
                    break
            
            # Final check
            if _is_valid(adj, target_n):
                edges = []
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if adj[i][j]:
                            edges.append([i, j])
                best_n = target_n
                best_edges = edges
                break
    
    return {'n': best_n, 'edges': best_edges}
