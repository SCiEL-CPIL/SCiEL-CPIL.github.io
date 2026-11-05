def _search_partition(n, k=5, deadline=None):
    elements = list(range(1, n))
    buckets = [set() for _ in range(k)]
    max_size = int(n ** 0.5) + 1
    
    def backtrack(idx):
        if deadline and time.time() > deadline:
            return False
        if idx == len(elements):
            return True
        
        e = elements[idx]
        # Order buckets by size to prune?
        # For now, simple iteration.
        for b in range(k):
            if idx == 0 and b != 0:
                continue
            
            if len(buckets[b]) + 1 > max_size:
                continue
            
            buckets[b].add(e)
            if _is_sidon(buckets[b], n):
                if backtrack(idx + 1):
                    return True
            buckets[b].remove(e)
        return False

    if backtrack(0):
        return buckets
    return None
