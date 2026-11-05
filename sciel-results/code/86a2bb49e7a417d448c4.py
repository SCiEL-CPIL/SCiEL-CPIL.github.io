     def local_search(c_list, c_set, cover_counts, unc_mask):
         improved = True
         while improved and time.time() < deadline - 5:
             improved = False
             # 1. Removal
             ...
             if not improved:
                 # 2. Swap
                 ...
         return c_list, c_set, cover_counts, unc_mask
     