   used = np.zeros(n_blocks, dtype=bool)
   uncovered = np.ones(num_words, dtype=np.uint64)
   uncovered[-1] &= last_word_mask
   
   selected = []
   
   while True:
       if not np.any(uncovered):
           break
       
       # Compute gains for all unused blocks
       # inter_all = block_bits & uncovered
       # This creates a large array.
       inter_all = np.bitwise_and(block_bits, uncovered)
       inter_bytes_all = inter_all.astype(np.uint8)
       gains = popcount_lookup[inter_bytes_all].sum(axis=1)
       
       # Mask used blocks
       gains[used] = -1
       
       best_idx = np.argmax(gains)
       best_gain = gains[best_idx]
       
       if best_gain <= 0:
           break
           
       selected.append(best_idx)
       used[best_idx] = True
       
       # Update uncovered
       uncovered = np.bitwise_and(uncovered, np.bitwise_not(block_bits[best_idx]))
       uncovered[-1] &= last_word_mask
   