<<<<<<< SEARCH
    # Ensure all outputs are in fast memory (blue) at the end
    for v in outputs:
        if v not in fast_mem:
            if v in slow_mem or v in computed:
                if len(fast_mem) >= fast_memory_budget:
                    candidates = [x for x in fast_mem if x not in outputs]
                    if candidates:
                        candidates.sort(key=lambda x: last_needed.get(x, 0))
                        evict = candidates[0]
                        ops.append(['delete', evict])
                        del fast_mem[evict]
                        ops.append(['store', evict])
                        slow_mem.add(evict)
                ops.append(['load', v])
                fast_mem[v] = True
=======
    # Ensure all outputs are in fast memory (blue) at the end
    for v in outputs:
        if v not in fast_mem:
            if v in slow_mem or v in computed:
                if len(fast_mem) >= fast_memory_budget:
                    candidates = [x for x in fast_mem if x not in outputs]
                    if candidates:
                        candidates.sort(key=lambda x: last_needed.get(x, 0))
                        evict = candidates[0]
                        ops.append(['store', evict])
                        del fast_mem[evict]
                        slow_mem.add(evict)
                ops.append(['load', v])
                fast_mem[v] = True
>>>>>>> REPLACE
