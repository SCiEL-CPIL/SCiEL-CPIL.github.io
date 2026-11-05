   if best_valid_adj_bits is None:
       # Fallback to n=27
       ...
       best_valid_adj_bits = adj_bits
       best_valid_n = n
       
   edges = _bitset_to_edges(best_valid_adj_bits, best_valid_n)
   return {'n': best_valid_n, 'edges': edges}
   