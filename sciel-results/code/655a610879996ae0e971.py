N = 100

# EVOLVE-BLOCK-START
def construct():
    # Create a byte string of N null bytes
    # Translate each null byte to '1'
    # Decode the resulting byte string to a standard Python string
    return bytes(N).translate(bytes.maketrans(b'\x00', b'1')).decode('ascii')
# EVOLVE-BLOCK-END
