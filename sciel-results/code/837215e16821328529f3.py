import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        if not edges:
            return []
        
        # Initialize paths
        paths = [[u, v] for u, v in edges]
        
        # Helper to check if a path is simple
        def is_simple(p):
            return len(p) == len(set(p))
            
        # Helper to merge two paths
        # Returns (new_path, index_a, index_b) or None
        def try_merge(pa, pb):
            # pa and pb are lists
            # Check if they share an endpoint
            ea = {pa[0], pa[-1]}
            eb = {pb[0], pb[-1]}
            
            common = ea.intersection(eb)
            if not common:
                return None
            
            v = list(common)[0]
            
            # Orient pa to end at v
            if pa[-1] != v:
                pa = pa[::-1]
            # Orient pb to start at v
            if pb[0] != v:
                pb = pb[::-1]
                
            # Now pa ends at v, pb starts at v
            # Merge
            new_path = pa + pb[1:]
            
            if is_simple(new_path):
                return new_path
            return None

        # Greedy merging
        # To avoid getting stuck, we can randomize the order of checks or restarts
        # But for small graphs, simple greedy is usually fine.
        # We will iterate until no more merges are possible.
        
        merged = True
        while merged:
            if time.time() - start_time > 100:
                break
            merged = False
            # Shuffle paths to avoid bias?
            # random.shuffle(paths) - might be expensive if done every loop.
            # Instead, just iterate.
            
            n_paths = len(paths)
            for i in range(n_paths):
                if merged:
                    continue
                for j in range(i + 1, n_paths):
                    if merged:
                        continue
                    new_path = try_merge(paths[i], paths[j])
                    if new_path:
                        # Replace i and j with new_path
                        # Remove j first to keep indices valid?
                        # paths[i] = new_path
                        # del paths[j]
                        paths[i] = new_path
                        del paths[j]
                        merged = True
                        break
                if merged:
                    break
                    
        return paths

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    for spec in specs:
        if time.time() - start_time > 110:
            # Fallback: return single edges
            edges = instance(*spec)
            results.append([[u, v] for u, v in edges])
            continue
            
        edges = instance(*spec)
        paths = decompose(edges)
        results.append(paths)
        
    return results
