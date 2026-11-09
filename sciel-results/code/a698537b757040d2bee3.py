     for v in outputs:
         if v in fast_mem:
             operations.append(['store', v])
             fast_mem.discard(v)
             transfers += 1
     