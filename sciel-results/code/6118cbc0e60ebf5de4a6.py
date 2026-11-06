   def apply_comp(state, i, j):
       mask_i = 1 << i
       mask_j = 1 << j
       # Extract bits
       # Since i < j, we can just use bit shifts
       # But we need to handle the whole array vectorized
       bit_i = (state >> i) & 1
       bit_j = (state >> j) & 1
       
       min_bit = bit_i & bit_j
       max_bit = bit_i | bit_j
       
       # Clear bits
       state &= ~(mask_i | mask_j)
       
       # Set bits
       # min goes to i, max goes to j
       state |= (min_bit << i)
       state |= (max_bit << j)
       
       return state
   