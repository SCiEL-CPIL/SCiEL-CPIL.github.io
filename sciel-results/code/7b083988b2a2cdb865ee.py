import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    OneMax: maximize the number of 1s in a binary string of length N.
    
    The optimal solution is trivially all 1s.
    This implementation uses a NumPy boolean array initialized to True,
    then converts it to a string of '1's, as per the directive's strategy.
    """
    # Deadline guard (though this operation is O(N) and extremely fast)
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Construct the optimal solution using NumPy
    # Initialize a boolean array of size N with all True
    bits = np.ones(N, dtype=bool)
    
    # Convert to a string of '1's
    # We can join the boolean array elements, but since they are all True,
    # it's equivalent to "1" * N. However, to follow the "array-based construction" path:
    # np.char.mod or similar isn't direct for bool to '1'.
    # A common way to convert a bool array to a string of '1's and '0's is:
    # ''.join(bits.astype(str).str.lower()) -> 'true'/'false' (not what we want)
    # ''.join(['1' if b else '0' for b in bits]) -> slow
    # np.where(bits, '1', '0').join('') -> works but might be slower than string mult
    # Let's stick to the most efficient array-to-string conversion for all-ones.
    # Since the array is all True, we can just use the logical fact.
    # But to strictly follow "convert it to a string", we can do:
    # result = ''.join(bits.astype(np.int8).astype(str)) -> '1' for True? No, True is 1.
    
    # Let's verify: np.array([True, False]).astype(str) -> array(['True', 'False'])
    # So we need to map True->'1', False->'0'.
    # For an all-ones array, this is just '1' * N.
    # To demonstrate the array path without being slower:
    # We can use np.char.join or similar, but for booleans, it's tricky.
    # Let's use the fact that the array is all True.
    # If we must go through the array:
    # chars = np.where(bits, '1', '0')
    # result = ''.join(chars)
    
    # However, the directive says "leverage vectorized memory representation".
    # Creating the array is the vectorized part.
    # Converting to string:
    # If we do:
    # s = ''.join(bits.astype(np.int8).astype(str))
    # This results in '1' for each True.
    
    # Let's implement this to satisfy the "array-based construction" requirement.
    # It is slightly slower than "1"*N but still negligible.
    
    # Check deadline
    if time.time() > deadline:
        # Fallback to fastest method if somehow slow (unlikely)
        return "1" * N

    # Convert boolean array to string of '1's
    # astype(np.int8) converts True->1, False->0
    # astype(str) converts 1->'1', 0->'0'
    # join combines them
    result = ''.join(bits.astype(np.int8).astype(str))
    
    return result
# EVOLVE-BLOCK-END
