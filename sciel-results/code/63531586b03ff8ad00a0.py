    def make_space():
        nonlocal fast_mem, ops
        if len(fast_mem) >= fast_memory_budget:
            evict_candidates = [x for x in fast_mem]
            if not evict_candidates:
                return False
            evict_candidates.sort(key=lambda x: crit[x])
            e = evict_candidates[0]
            ops.append(["delete", e])
            fast_mem.remove(e)
        return True

    def ensure_in_fast(v):
        nonlocal fast_mem, slow_mem, ops
        
        if v in fast_mem:
            return True
        
        if v in slow_mem:
            if not make_space():
                return False
            ops.append(["load", v])
            fast_mem.add(v)
            return True
        
        # Not in fast or slow. Must recompute.
        if v in inputs:
            return False
        
        # Recompute v
        for p in parents.get(v, []):
            if not ensure_in_fast(p):
                return False
        
        if not make_space():
            return False
            
        ops.append(["compute", v])
        fast_mem.add(v)
        return True
