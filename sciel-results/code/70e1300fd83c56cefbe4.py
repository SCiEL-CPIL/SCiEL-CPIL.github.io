   def update_img(img, dest, q, old_mask, new_mask):
       # Update img[S] for all S containing q
       # dest is already updated at index q? 
       # Let's say dest[q] is already new_mask.
       mask_q = 1 << q
       for S in range(1 << N):
           if S & mask_q:
               # Recompute img[S]
               m = 0
               # Iterate bits in S
               s = S
               while s:
                   lsb = s & -s
                   i = (lsb.bit_length() - 1)
                   m |= dest[i]
                   s ^= lsb
               img[S] = m
   