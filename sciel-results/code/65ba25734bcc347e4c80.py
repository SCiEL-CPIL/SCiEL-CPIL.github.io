      ops.append(['store', evict_target])
      ops.append(['delete', evict_target])
      in_fast.discard(evict_target)
      fast_count -= 1
      