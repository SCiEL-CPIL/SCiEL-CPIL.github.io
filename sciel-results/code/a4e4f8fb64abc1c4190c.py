   def get_used_pairs(triples):
       used = set()
       for t in triples:
           for p in triple_pairs(t):
               used.add(p)
       return used
   