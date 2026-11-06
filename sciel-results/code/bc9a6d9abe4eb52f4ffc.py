      # Before computing v
      ps = parents.get(v, [])
      protected = set(ps) # Parents needed for this compute
      
      # Ensure all parents are in fast_mem
      missing = [p for p in ps if p not in fast_mem]
      
      # We need to make room for 'missing'
      # Current used slots: len(fast_mem)
      # Slots needed: len(fast_mem) + len(missing)
      # If > budget, evict (len(fast_mem) + len(missing) - budget) items
      
      needed_slots = len(fast_mem) + len(missing)
      if needed_slots > fast_memory_budget:
          num_evict = needed_slots - fast_memory_budget
          # Find evictable vertices
          # Candidates: in fast_mem, not in protected, not in must_be_blue
          
          candidates = [v for v in fast_mem if v not in protected and v not in must_be_blue]
          # Sort by priority: lowest remaining_uses, then earliest topo
          candidates.sort(key=lambda x: (remaining_uses[x], -topo_pos.get(x, 0)))
          
          if len(candidates) < num_evict:
              # Critical Error: Not enough evictable vertices.
              # This happens if budget < fan-in or if we are holding too many outputs/parents.
              # Since we can't solve it, we might just proceed and hope for the best, or raise.
              # For robustness, evict as many as possible.
              pass
          
          for i in range(min(num_evict, len(candidates))):
              ev = candidates[i]
              ops.append(['store', ev])
              fast_mem.discard(ev)
              
          # If we still don't have enough room, we are in trouble.
          # But we proceed to load missing parents.
          
      for p in missing:
          ops.append(['load', p])
          fast_mem.add(p)
      
      # Now all parents should be in fast_mem (assuming budget was sufficient)
      ops.append(['compute', v])
      fast_mem.add(v)
      
      # Post-compute cleanup
      for p in ps:
          remaining_uses[p] -= 1
          if remaining_uses[p] == 0 and p not in must_be_blue and p in fast_mem:
              # Can we delete p?
              # Only if it's not needed for future computations?
              # remaining_uses == 0 means no future children need it.
              # So yes, delete to free space.
              ops.append(['delete', p])
              fast_mem.discard(p)
      