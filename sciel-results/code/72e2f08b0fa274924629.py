   def try_delete(word, n):
       # word is list of ints
       # all_perms is a set of tuples
       # We need to check if all perms are in windows(word)
       
       # Optimization: 
       # We can maintain a count of how many times each perm appears.
       # But updating counts on deletion is hard due to shifts.
       
       # So we just rebuild.
       # To speed up, we can use a bitset if possible? No, 5040 perms.
       
       # Let's just do the rebuild.
       # N is ~6000.
       # Rebuild: 6000 * 7 = 42000 ops.
       # 6000 iterations * 42000 = 2.5e8 ops.
       # This is borderline.
       
       # We can break early if we find a missing perm.
       
       pass
   