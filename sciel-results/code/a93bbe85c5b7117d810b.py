import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    OneMax: maximize the number of 1s in a binary string of length N.
    
    The optimal solution is trivially all 1s.
    This implementation uses a mutable bytearray initialized with ASCII '1's,
    then decodes it to a string. This leverages mutable buffer allocation
    as a distinct construction path from immutable string multiplication
    or NumPy array-based methods.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    if time.time() > deadline:
        return "1" * N

    # Initialize a bytearray with N bytes, each set to the ASCII code for '1' (49)
    # bytearray(N) initializes with zeros, so we fill it with 49.
    # Alternatively, bytearray(b'1' * N) creates the buffer directly.
    # bytearray(N) then filling is slightly more explicit about the buffer nature.
    # But b'1' * N is a bytes object, which is immutable. 
    # The directive asks for "bytearray initialized with 100 ASCII '1' characters".
    
    # Method: Create a bytes object of '1' repeated N times, then convert to bytearray?
    # Or just create the bytearray and fill it?
    # bytearray(b'1' * N) is efficient.
    
    buf = bytearray(b'1' * N)
    
    # Decode to string
    result = buf.decode('ascii')
    
    return result
# EVOLVE-BLOCK-END
