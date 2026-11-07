import struct

# Constants
NUM_ITEMS = 400
STATE_SIZE = 1024

def update(state, item):
    """
    Update the state by incrementing the exact count for the given item.
    State layout:
    - Bytes 0..399: exact counts for items 0..399 (uint8 each)
    - Bytes 400..1023: unused (zeros)
    """
    # Convert bytes to bytearray for mutable access
    ba = bytearray(state)
    
    # Increment the count for this item
    current = ba[item]
    # Cap at 255 to avoid overflow issues (though counts won't exceed 1024/400 avg ~2.5)
    # Actually max possible is 1024, but uint8 max is 255.
    # Wait, the problem says 3 streams of 1024 items.
    # Stream 1: uniform 0..399. Expected count per item = 1024/400 = 2.56. Max likely < 10.
    # Stream 2: Zipf(1.6) mod 400. Item 0 will be very frequent.
    # Stream 3: 512 uniform 0..19, 512 uniform 150..399.
    #     Items 0..19: 512/20 = 25.6 expected.
    #     Items 150..399: 512/250 = 2.048 expected.
    #     Items 20..149: 0 from stream 3.
    #
    # Total counts:
    # Item 0: ~2.56 (S1) + High (S2) + 25.6 (S3).
    # Zipf(1.6) on 1..400 mapped to 0..399?
    # "Zipf(1.6) modulo 400". Usually Zipf generates 1..N.
    # If item is generated as zipf(1.6) % 400.
    # Zipf(1.6) probability for rank k: 1/(k^1.6 * H(400, 1.6)).
    # H(400, 1.6) approx integral x^-1.6 dx from 1 to 400 + 0.5
    # integral = [x^-0.5 / -0.5] = -2/sqrt(x).
    # -2/sqrt(400) - (-2/sqrt(1)) = -2/20 + 2 = 1.9.
    # So H ~ 1.9 + 0.5 = 2.4?
    # P(rank 1) = 1/(1 * 2.4) ~ 0.41.
    # So item 0 (if rank 1 maps to 0) gets ~41% of 1024 = ~420 items.
    # 420 > 255. Overflow!
    
    # We need a better representation for high-frequency items.
    # Let's use the 1024 bytes more wisely.
    # We have 400 items.
    # Let's use 16-bit counters? 400 * 2 = 800 bytes.
    # Remaining 224 bytes can be used for something else or just padding.
    # Max count per item is 1024 (if all items are the same).
    # 16-bit unsigned can hold up to 65535.
    # This is safe and simple.
    
    # Re-layout:
    # Bytes 0..799: 400 uint16 counters (little-endian)
    # Bytes 800..1023: padding
    
    # Update the 16-bit counter
    offset = item * 2
    val = struct.unpack_from('<H', ba, offset)[0]
    val += 1
    struct.pack_into('<H', ba, offset, val)
    
    return bytes(ba)

def query(state, item):
    """
    Query the exact count for the given item.
    """
    offset = item * 2
    val = struct.unpack_from('<H', state, offset)[0]
    return float(val)
