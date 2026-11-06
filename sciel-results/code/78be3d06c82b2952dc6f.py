   def is_valid(network):
       inputs = np.arange(2**18, dtype=np.uint32)
       outputs = simulate(network, inputs)
       # For each input, count ones
       # We can precompute the count of ones for all 2^18 numbers?
       # Or compute on the fly.
       # Popcount for uint32 array in NumPy?
       # Not directly vectorized in standard NumPy without loops or LUT.
       # We can use a LUT for bytes.
       
       # Alternative: Check if outputs are sorted.
       # An output is sorted if for all k < l, if output[k]=1 then output[l]=1?
       # No, sorted means 0s then 1s.
       # So for each row, there is at most one transition 0->1.
       
       # Faster check:
       # For each wire i, output[i] must be 0 if i < count_ones(input) and 1 if i >= count_ones(input).
       
       # Let's precompute the target outputs.
       # target[k] is the sorted version of inputs[k].
       # Since inputs[k] is just the integer k, the number of ones is popcount(k).
       # The sorted output has popcount(k) ones at the highest positions.
       # So target[k] = ((1 << popcount(k)) - 1) << (18 - popcount(k)).
       
       # We need popcount for 262k numbers.
       # We can do this with a LUT of size 256.
       def popcount_array(arr):
           # arr is uint32
           # Split into bytes
           # This might be slow in pure Python/NumPy without C.
           # Alternative:
           # We can just check if the output array is "sorted" in the bitwise sense.
           # A vector is sorted if (output >> i) & 1 >= (output >> j) & 1 for all i < j?
           # No, 0 < 1. So we need output[i] <= output[j] for i < j.
           # So for each adjacent pair (i, i+1), we need output[i] <= output[i+1].
           # If this holds for all i, the vector is sorted.
           
           # Let's check adjacent bits.
           # For i in 0..16:
           #   bit_i = (outputs >> i) & 1
           #   bit_ip1 = (outputs >> (i+1)) & 1
           #   if any(bit_i > bit_ip1): return False
           
           for i in range(17):
               bit_i = (outputs >> i) & 1
               bit_ip1 = (outputs >> (i+1)) & 1
               if np.any(bit_i > bit_ip1):
                   return False
           return True
   